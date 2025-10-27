#!/usr/bin/env python

import os
import time
import numpy as np
import mrcfile
from os.path import dirname, abspath
from segVesicle.models import resunet3D as models


def _segment_core(predict_patch, tomopath, patch_size=192):
    with mrcfile.open(tomopath) as m:
        dataArray = m.data
    pcrop = 48  # how many pixels to crop from border
    patch_size = min(patch_size, (dataArray.shape[0] + 2 * pcrop) // 8 * 8)
    Ncl = 2

    percentile_99_5 = np.percentile(dataArray, 99.5)
    percentile_00_5 = np.percentile(dataArray, 0.5)
    dataArray = np.clip(dataArray, percentile_00_5, percentile_99_5)

    dataArray = (dataArray[:] - np.mean(dataArray[:])) / np.std(dataArray[:])  # normalize
    dataArray = np.pad(dataArray, pcrop, mode='reflect')  # reflect pad
    dim = dataArray.shape
    l = int(patch_size / 2)
    lcrop = int(l - pcrop)
    step = int(patch_size - 2 * pcrop)
    # Get patch centers:
    pcenterZ = list(range(l, dim[0] - l, step))
    pcenterY = list(range(l, dim[1] - l, step))
    pcenterX = list(range(l, dim[2] - l, step))
    # If there are still few pixels at the end:
    if pcenterX[-1] < dim[2] - l:
        pcenterX.append(dim[2] - l)
    if pcenterY[-1] < dim[1] - l:
        pcenterY.append(dim[1] - l)
    if pcenterZ[-1] < dim[0] - l:
        pcenterZ.append(dim[0] - l)
    Npatch = len(pcenterX) * len(pcenterY) * len(pcenterZ)

    print('Data array is divided in ' + str(Npatch) + ' patches ...')
    # ---------------------------------------------------------------
    # Process data in patches:
    start = time.time()

    predArray = np.zeros(dim + (Ncl,), dtype=np.float16)
    normArray = np.zeros(dim, dtype=np.int8)
    patchCount = 1
    for x in pcenterX:
        for y in pcenterY:
            for z in pcenterZ:
                print('Segmenting patch ' + str(patchCount) + ' / ' + str(Npatch) + ' ...')
                patch = dataArray[z - l:z + l, y - l:y + l, x - l:x + l]
                patch = np.reshape(patch, (1, patch_size, patch_size, patch_size, 1))
                pred = predict_patch(patch)
                if pred.ndim == 4:
                    pred = np.expand_dims(pred, axis=0)
                predArray[z - lcrop:z + lcrop, y - lcrop:y + lcrop, x - lcrop:x + lcrop, :] += np.float16(
                    pred[0, l - lcrop:l + lcrop, l - lcrop:l + lcrop, l - lcrop:l + lcrop, :]
                )
                normArray[z - lcrop:z + lcrop, y - lcrop:y + lcrop, x - lcrop:x + lcrop] += np.ones(
                    (patch_size - 2 * pcrop, patch_size - 2 * pcrop, patch_size - 2 * pcrop), dtype=np.int8
                )
                patchCount += 1
    normArray[normArray == 0] = 1

    # Normalize overlaping regions:
    for C in range(0, Ncl):
        predArray[:, :, :, C] = predArray[:, :, :, C] / normArray
    end = time.time()
    print("Model took {} seconds to predict".format(int(end - start)))
    predArray = predArray[pcrop:-pcrop, pcrop:-pcrop, pcrop:-pcrop, :]  # unpad

    labelmap = np.int8(np.argmax(predArray, 3))
    return labelmap


def segment(path_weights, tomopath, patch_size=192):
    Ncl = 2
    net = models.my_model(patch_size, Ncl)
    net.load_weights(path_weights)

    def _predict(patch):
        return net.predict(patch, batch_size=10)

    return _segment_core(_predict, tomopath, patch_size)


def _select_providers(ort, requested=None):
    available = ort.get_available_providers()
    if requested:
        ordered = [p for p in requested if p in available]
        if ordered:
            return ordered
    preferred = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ordered = [p for p in preferred if p in available]
    return ordered if ordered else available


def segment_onnx(path_weights, tomopath, patch_size=192, providers=None):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for ONNX inference, please install it via pip or conda."
        ) from exc

    session = ort.InferenceSession(path_weights, providers=_select_providers(ort, providers))
    input_name = session.get_inputs()[0].name

    def _predict(patch):
        patch = patch.astype(np.float32, copy=False)
        return session.run(None, {input_name: patch})[0]

    return _segment_core(_predict, tomopath, patch_size)


import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--tomo', type=str, default=None, help='tomo file')
parser.add_argument('--tomo_file', type=str, default=None, help='the isonet_corrected tomo file')
parser.add_argument('--iso_path', type=str, default=None, help='the isonet_model.h5 file path')
parser.add_argument('--tomo_deconv_file', type=str, default=None, help='the wbp_deconvolution tomo file')
parser.add_argument('--dec_path', type=str, default=None, help='the dec_model.h5 file path')
parser.add_argument('--mask_file', type=str, default=None, help='the output vesicle segment file name')
parser.add_argument('--gpuID', type=str, default=0, help='The gpuID to used during the training. e.g 0,1,2,3.')
parser.add_argument('--mode', type=int, default=0, help='0 for double models, 1 for only iso-model, 2 for only dec-model')
parser.add_argument(
    '--inference_engine',
    type=str,
    choices=['keras', 'onnx'],
    default='onnx',
    help='Select Keras (.h5 weights) or ONNX (.onnx weights) inference backend.',
)
parser.add_argument(
    '--onnx_providers',
    type=str,
    default="CUDAExecutionProvider",
    help='Comma separated ONNXRuntime providers (e.g. "CUDAExecutionProvider,CPUExecutionProvider").',
)

args = parser.parse_args()

# set some default files 
if args.tomo_file is None:
    args.tomo_file = args.tomo + '_wbp_corrected.mrc'
if args.tomo_deconv_file is None:
    args.tomo_deconv_file = args.tomo + '_wbp_dec.mrc'
if args.mask_file is None:
    args.mask_file = args.tomo + '_segment.mrc'

segVesicleHome = dirname(abspath(__file__))
segVesicleHome = os.path.split(segVesicleHome)[0]+'/'

args.gpuID = str(args.gpuID)
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuID

def _default_weight(engine, model_idx):
    suffix = 'onnx' if engine == 'onnx' else 'h5'
    return segVesicleHome + f'pretrained/vesicle_seg_model_{model_idx}.{suffix}'


path_weights1 = args.iso_path
if args.iso_path is None:
    path_weights1 = _default_weight(args.inference_engine, 1)
if not os.path.exists(path_weights1):
    raise ValueError(f"Iso-Model file {path_weights1} does not exist.")

path_weights2 = args.dec_path
if args.dec_path is None:
    path_weights2 = _default_weight(args.inference_engine, 2)
if not os.path.exists(path_weights2):
    raise ValueError(f"Dec-Model file {path_weights2} does not exist.")

tomopath1 = args.tomo_file
tomopath2=args.tomo_deconv_file

providers = None
if args.onnx_providers:
    providers = [p.strip() for p in args.onnx_providers.split(',') if p.strip()]

segment_fn = segment_onnx if args.inference_engine == 'onnx' else segment


def _run_segmentation(weights_path, tomo_path):
    if args.inference_engine == 'onnx':
        return segment_fn(weights_path, tomo_path, providers=providers)
    return segment_fn(weights_path, tomo_path)


if args.mode == 1:
    seg1 = _run_segmentation(path_weights1, tomopath1)
    seg2 = 0
elif args.mode == 2:
    seg1 = 0
    seg2 = _run_segmentation(path_weights2, tomopath2)
elif args.mode == 0:
    seg1 = _run_segmentation(path_weights1, tomopath1)
    seg2 = _run_segmentation(path_weights2, tomopath2)
else:
    raise ValueError("Wrong mode!")

with mrcfile.new(args.mask_file,overwrite=True) as m:
    m.set_data(np.sign(seg1+seg2).astype(np.int8))
