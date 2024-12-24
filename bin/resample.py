#!/usr/bin/env python
import numpy as np
import mrcfile
import fire
from scipy.ndimage import zoom
from util.io import get_tomo



def generate_new_tomo(out_data, out_name, outspacing):
    with mrcfile.new(out_name, overwrite=True) as m:
        m.set_data(out_data)
        m.voxel_size = [outspacing, outspacing, outspacing]


def resample_image(tomo, pixel_size, out_name=None, outspacing=17.142):

    original_data = get_tomo(tomo)
    
    original_spacing = [pixel_size, pixel_size, pixel_size]
    out_spacing = [outspacing, outspacing, outspacing]
    scale_factors = [
        original_spacing[0] / out_spacing[0],
        original_spacing[1] / out_spacing[1],
        original_spacing[2] / out_spacing[2]
    ]
    
    # resample
    out_data = zoom(original_data, zoom=scale_factors, order=1)
    
    if out_name is None:
        out_name = tomo[:-4] + '-resample' + tomo[-4:]
    
    generate_new_tomo(out_data, out_name, outspacing)

    return out_data


def measure(tomo, pixel_size, outspacing=17.142):
    
    tomo_data = get_tomo(tomo)
    original_spacing = [pixel_size, pixel_size, pixel_size]
    original_spacing = [pixel_size, pixel_size, pixel_size]
    
    original_size = tomo_data.shape
    output_spacing = [outspacing, outspacing, outspacing]
    out_size = [
        int(np.round(original_size[0] * original_spacing[0] / output_spacing[0])),
        int(np.round(original_size[1] * original_spacing[1] / output_spacing[1])),
        int(np.round(original_size[2] * original_spacing[2] / output_spacing[2]))
    ]

    return [original_spacing, original_size, output_spacing, out_size]


def main(tomo, pixel_size, outname, outspacing=17.142):
    resample_image(tomo, pixel_size, outname, outspacing=outspacing)


if __name__ == "__main__":
    fire.Fire(main)