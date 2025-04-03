#!/usr/bin/env python

import os
import numpy as np
import mrcfile
import time
from os.path import dirname, abspath
from segVesicle.models import resunet3D as models

def segment(path_weights,tomopath,patch_size=192):
    with mrcfile.open(tomopath) as m:
        dataArray=m.data
    pcrop = 48  # how many pixels to crop from border
    patch_size = min(patch_size, (dataArray.shape[0]+2*pcrop)//8*8)
    Ncl=2
    #build network
    net=models.my_model(patch_size,Ncl)
    net.load_weights(path_weights)
    
    percentile_99_5 = np.percentile(dataArray, 99.5)
    percentile_00_5 = np.percentile(dataArray, 00.5)
    dataArray=np.clip(dataArray, percentile_00_5, percentile_99_5)

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
                patch = np.reshape(patch, (1, patch_size, patch_size, patch_size, 1))  # reshape for keras [batch,x,y,z,channel]
                pred = net.predict(patch, batch_size=10)
                predArray[z-lcrop:z+lcrop, y-lcrop:y+lcrop, x-lcrop:x+lcrop, :] += np.float16(pred[0,l - lcrop:l + lcrop,l - lcrop:l + lcrop,l - lcrop:l + lcrop,:])
                normArray[z-lcrop:z+lcrop, y-lcrop:y+lcrop, x-lcrop:x+lcrop]+=np.ones((patch_size-2*pcrop, patch_size-2*pcrop, patch_size-2*pcrop), dtype=np.int8)
                patchCount += 1
    normArray[normArray==0]=1

    # Normalize overlaping regions:
    for C in range(0, Ncl):
        predArray[:, :, :, C] = predArray[:, :, :, C] / normArray
    end = time.time()
    print("Model took {} seconds to predict".format(int(end - start)))
    predArray = predArray[pcrop:-pcrop, pcrop:-pcrop, pcrop:-pcrop, :]  # unpad

    labelmap = np.int8( np.argmax(predArray,3) )
    return labelmap




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

path_weights1 = args.iso_path
if args.iso_path is None:
    path_weights1 = segVesicleHome + 'pretrained/vesicle_seg_model_1.h5'
if not os.path.exists(path_weights1):
    raise ValueError(f"Iso-Model file {path_weights1} does not exist.")

path_weights2 = args.dec_path
if args.dec_path is None:
    path_weights2 = segVesicleHome + 'pretrained/vesicle_seg_model_2.h5'
if not os.path.exists(path_weights2):
    raise ValueError(f"Dec-Model file {path_weights2} does not exist.")

tomopath1 = args.tomo_file
tomopath2=args.tomo_deconv_file

if args.mode == 1:
    seg1 = segment(path_weights1,tomopath1)
    seg2 = 0
elif args.mode == 2:
    seg1 = 0
    seg2 = segment(path_weights2,tomopath2)
elif args.mode == 0:
    seg1 = segment(path_weights1,tomopath1)
    seg2 = segment(path_weights2,tomopath2)
else:
    raise ValueError("Wrong mode!")

with mrcfile.new(args.mask_file,overwrite=True) as m:
    m.set_data(np.sign(seg1+seg2).astype(np.int8))

