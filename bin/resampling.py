#!/usr/bin/env python3

# This is an efficient program for resampling tomogram based on python-simpleITK
# Author: Zhenhang Lu
# Version: 2022/03/04

import SimpleITK as sitk
import numpy as np
import mrcfile
import argparse
import multiprocessing



# def prepare_resample(tomo, pixel_size, outspacing=17.142):
#     sitk_tomo = sitk.ReadImage(tomo)
#     # if sitk_tomo.GetSpacing()[0] == 1:
#     #     ratio = outspacing / pixel_size
#     # else:
#     #     ratio = outspacing / sitk_tomo.GetSpacing()[0]
#     ratio = outspacing / pixel_size
#     out_spacing = [ratio, ratio, ratio]
#     return out_spacing


def generate_new_tomo(resample_tomo, out_name, outspacing):
    resampletomo_data = sitk.GetArrayFromImage(resample_tomo)
    with mrcfile.new(out_name, overwrite=True) as m:
        m.set_data(resampletomo_data)
        m.voxel_size = [outspacing, outspacing, outspacing]


def resample_image(tomo, pixel_size, outspacing=17.142):

    tomo_sitk = sitk.ReadImage(tomo)
    #out_spacing = prepare_resample(tomo, pixel_size, outspacing)
    out_spacing = [outspacing, outspacing, outspacing]
    # original_spacing = tomo_sitk.GetSpacing()
    original_spacing = [pixel_size, pixel_size, pixel_size]
    original_size = tomo_sitk.GetSize()
    # if original_spacing[0] != 1:
    #     out_spacing = out_spacing
    # else:
    #     out_spacing = [0.942, 0.942, 0.942]
    out_spacing = [outspacing, outspacing, outspacing]
    out_size = [
        int(np.round(original_size[0] * original_spacing[0] / out_spacing[0])),
        int(np.round(original_size[1] * original_spacing[1] / out_spacing[1])),
        int(np.round(original_size[2] * original_spacing[2] / out_spacing[2]))
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(tomo_sitk.GetDirection())
    resample.SetOutputOrigin(tomo_sitk.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(tomo_sitk.GetPixelIDValue())
    # resample.SetInterpolator(sitk.sitkBSpline)
    resample.SetInterpolator(sitk.sitkLinear)

    resample_tomo = resample.Execute(tomo_sitk)
        
    out_name = tomo.split('.')[0] + '-resample.' + tomo.split('.')[1]
    generate_new_tomo(resample_tomo, out_name, outspacing)

    return resample_tomo


def measure(tomo, pixel_size, outspacing=17.142):
    sitk_tomo = sitk.ReadImage(tomo)
    # original_spacing = sitk_tomo.GetSpacing()
    original_spacing = [pixel_size, pixel_size, pixel_size]
    # if original_spacing[0] != 1:
    #     original_spacing = original_spacing
    # else:
    #     original_spacing = [pixel_size, pixel_size, pixel_size]
    original_spacing = [pixel_size, pixel_size, pixel_size]
    original_size = sitk_tomo.GetSize()
    #output_spacing = prepare_resample(tomo, pixel_size, outspacing)
    output_spacing = [outspacing, outspacing, outspacing]
    out_size = [
        int(np.round(original_size[0] * original_spacing[0] / output_spacing[0])),
        int(np.round(original_size[1] * original_spacing[1] / output_spacing[1])),
        int(np.round(original_size[2] * original_spacing[2] / output_spacing[2]))
    ]
    #out_spacing = [outspacing, outspacing, outspacing]
    return [original_spacing, original_size, output_spacing, out_size]


# def process_tomo(args):
#     tomo, pixel_size, outspacing = args
#     return resample_image(tomo, pixel_size, outspacing)

# def main(tomos, pixel_size, outspacing=17.142):
#     args = [(tomo, pixel_size, outspacing) for tomo in tomos]
#     # with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
#     #     resample_tomos = pool.map(process_tomo, args)
#     with multiprocessing.Pool() as pool:
#         resample_tomos = pool.map(process_tomo, args)
#     return resample_tomos


if __name__ == '__main__':  

    parser = argparse.ArgumentParser(description='set output spacing and original tomo')
    parser.add_argument('--tomo', type=str, default=None, help='original tomo')
    parser.add_argument('--outspacing', type=float, default=17.141, help='output voxel spacing')
    args = parser.parse_args()

    resample_tomo = resample_image(args.tomo, args.outspacing)



