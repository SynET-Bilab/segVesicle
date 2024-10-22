#!/usr/bin/env python3

import sys
import json
import mrcfile
import logging
import numpy as np

from tqdm import tqdm
from scipy import ndimage
from scipy.sparse import csr_matrix
from skimage.measure import label, regionprops
from skimage.morphology import opening, closing, erosion, dilation, remove_small_objects
from skimage.morphology import cube, ball, disk, square

from morph_bak import get_indices_sparse
from segVesicle.bin.boundary_mask import boundary_mask
from segVesicle.utils import make_ellipsoid as mk
from segVesicle.bin.ellipsoid import ellipsoid_fit as ef

def one_label_fit(data):
    l = label(data)    
    circle = dilation(l,ball(1)) - erosion(l,ball(1))
    idx=get_indices_sparse(circle)
    vesicle_points=np.swapaxes(np.array(idx[1]),0,1)
    [center, evecs, radii]=ef.ellipsoid_fit(vesicle_points)
    return center, evecs, radii




def labels2json(labels_data,jsonfile):
    results = []
    IDset = np.unique(labels_data)[1:]
    data_pad = np.pad(labels_data,30,'constant',constant_values= 0)
    for ID in tqdm(IDset, file=sys.stdout):
        if ID > 20000: continue
        data = data_pad.copy()
        data[data_pad!=ID] = 0
        data[data_pad==ID] = 1
        if np.sum(data) < 400: continue

        # # l = label(data)
        # # regions = regionprops(l)
        # # #一般是一个[0,1]的球，若不是则选最大体积的
        # # if np.max(l) > 1:
        # #     max_area = 0
        # #     max_label = 0
        # #     for i in range(len(regions)):
        # #         area = regions[i].area
        # #         la = i + 1
        # #         if area > max_area:
        # #             max_area = area
        # #             max_label = la
        # #     l[l!=max_label] = 0
        # #     l[l>0] = 1
        # #     if np.sum(l) < 400: continue

        # this region为一个[0,1]的连通体球
        region = regionprops(data)[0]       
        center_region = np.array(region.centroid).astype(np.int16)
        data_cube = data_pad[center_region[0]-25:center_region[0]+25, center_region[1]-25:center_region[1]+25,center_region[2]-25:center_region[2]+25]

        center, evecs, radii = one_label_fit(data_cube)
        # v_center 为原label mrc下的坐标
        v_center = center_region -25 + center - 30
        #label value 为 ID
        info={'name':str(ID),'center':v_center.tolist(),'radii':radii.tolist(),'evecs':evecs.tolist(),'CCF':'1'}
        results.append(info)
    #vesicle_info={'pixelsize':str(17.14),'number':str(len(results)),'vesicles':results}
    vesicle_info={'vesicles':results}
    if jsonfile is not None:    
        with open(jsonfile,"w") as out:
            json.dump(vesicle_info,out)
    return vesicle_info



if __name__ == "__main__":
    
    import argparse
    import time
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--tomo', type=str, default=None, help='tomo file')
    parser.add_argument('--label', type=str, default=None, help='label mrc file')
    parser.add_argument('--jsonfile', type=str, default=None, help='output json file')

    args = parser.parse_args()
    if args.label is None:
        args.label = args.tomo + '_label_vesicle.mrc'
    if args.jsonfile is None:
        args.jsonfile = args.tomo + '_vesicle_all.json'
    
    with mrcfile.open(args.label) as m:
        labeldata = m.data.astype(np.int16)
    t1 = time.time()
    vesicle_info = labels2json(labeldata,args.jsonfile)
    print(f'done json generating, cost {time.time()-t1} s')
