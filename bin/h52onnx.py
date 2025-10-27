#!/usr/bin/env python

import argparse
from pathlib import Path

import tensorflow as tf
import tf2onnx

from segVesicle.models import resunet3D as models

segVesicleHome = Path(__file__).resolve().parent.parent

DEFAULT_MODELS = [
    (
        segVesicleHome / 'pretrained' / 'vesicle_seg_model_1.h5',
        segVesicleHome / 'pretrained' / 'vesicle_seg_model_1.onnx',
    ),
    (
        segVesicleHome / 'pretrained' / 'vesicle_seg_model_2.h5',
        segVesicleHome / 'pretrained' / 'vesicle_seg_model_2.onnx',
    ),
]


def convert(weights_path: Path, output_path: Path, patch_size: int = 192):
    print(f'Converting {weights_path} -> {output_path}')
    if not weights_path.exists():
        raise FileNotFoundError(f'Weights file {weights_path} does not exist.')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = models.my_model(patch_size, 2)
    model.load_weights(str(weights_path))

    input_sig = (tf.TensorSpec((1, patch_size, patch_size, patch_size, 1), tf.float32, name='input'),)
    tf2onnx.convert.from_keras(model, input_signature=input_sig, output_path=str(output_path), opset=13)
    tf.keras.backend.clear_session()
    print(f'Saved ONNX model to {output_path}')


def parse_pairs(raw_pairs):
    if not raw_pairs:
        return DEFAULT_MODELS
    pairs = []
    for item in raw_pairs:
        if ':' in item:
            src, dst = item.split(':', 1)
            pairs.append((Path(src), Path(dst)))
        else:
            src = Path(item)
            pairs.append((src, src.with_suffix('.onnx')))
    return pairs


def main():
    parser = argparse.ArgumentParser(description='Convert vesicle segmentation Keras models (.h5) to ONNX format.')
    parser.add_argument(
        '--models',
        nargs='*',
        default=None,
        help='Optional list of conversion pairs. Use "source.h5:target.onnx" or just "source.h5" (default output = .onnx).',
    )
    parser.add_argument('--patch-size', type=int, default=192, help='Patch size used when building the network.')
    args = parser.parse_args()

    pairs = parse_pairs(args.models)
    for src, dst in pairs:
        convert(Path(src), Path(dst), patch_size=args.patch_size)


if __name__ == '__main__':
    main()
