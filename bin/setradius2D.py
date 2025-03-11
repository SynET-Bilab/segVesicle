#!/usr/bin/env python3

import sys
import json
import mrcfile
import logging
import numpy as np
import multiprocessing
from tqdm import tqdm
from scipy import ndimage
from scipy.sparse import csr_matrix
from skimage.measure import label
from skimage.morphology import opening, closing, erosion, dilation, remove_small_objects
from skimage.morphology import cube, ball, disk, square

from segVesicle.bin.boundary_mask import boundary_mask
from segVesicle.utils import make_ellipsoid as mk
from segVesicle.bin.ellipsoid import ellipsoid_fit as ef
from morph import template_2d, CCF, density_fit_2d, dis



def set_2D_radius(input_json_file,output_file, mrc_data):
    with open(input_json_file) as f:
        ves = json.load(f)
    vesicle_info = ves['vesicles']
    for info in tqdm((vesicle_info), file=sys.stdout):
    # for info in (vesicle_info[75],):
        center = np.array(info['center'])
        radius3D = np.array(info['radii'])
        radius_init = np.mean(radius3D)
        z_init = np.round(center[0]).astype(np.int16) #z,y,x
        y_init = np.round(center[1]).astype(np.int16)
        x_init = np.round(center[2]).astype(np.int16)
        z_range = range(z_init - 1, z_init +2)
        r_ma = 0
        for z in z_range:
            center_z = np.array([z, y_init, x_init])
            center_fit, evecs_fit, radii_fit, ccf = density_fit_2d(mrc_data, center_z, radius_init)
            if radii_fit is not None:

                r_z = 0.5 * (radii_fit[1] + radii_fit[2])
                if r_z > r_ma:
                    r_ma = r_z
                    info['center2D'] = center_fit.tolist()
                    info['radius2D'] = [radii_fit[1], radii_fit[2]]
                    info['evecs2D'] = evecs_fit[-2:,-2:].tolist()
        with open(output_file,'w') as f:
            json.dump(ves,f)


if __name__ == "__main__":

    import argparse
    import os
    import time
    import mrcfile
    import numpy as np

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--tomo_file', type=str, required=True, help='the isonet_corrected tomo file')
    parser.add_argument('--json_file', type=str, help='input and output json file')

    args = parser.parse_args()

    if not os.path.exists(args.tomo_file):
        raise ValueError(f"Tomo file {args.tomo_file} does not exist.")
    if not os.path.exists(args.json_file):
        raise ValueError(f"Tomo file {args.json_file} does not exist.")

    output_jsonfile = args.json_file.replace(".json", "_2D.json")


    with mrcfile.open(args.tomo_file) as m:
        mrc_data = m.data

    set_2D_radius(args.json_file, output_jsonfile, mrc_data)