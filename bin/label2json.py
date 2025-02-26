#!/usr/bin/env python3
import os
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

def one_label_2d_fit(data):
    l = label(data)
    cube_ = np.zeros((l.shape[0],l.shape[0],l.shape[0]))
    circle = dilation(l,disk(1)) - erosion(l,disk(1))
    cube_[cube_.shape[0]//2] = circle
    cloud=np.where(cube_>0)
    x = np.asarray(cloud[2])
    y = np.asarray(cloud[1])
    z = 0
    [center, evecs, radii]=ef.ellipse_fit(x,y,z)
    return center, evecs, radii

def labels2json(labels_data,jsonfile):
    results = []
    IDset = np.unique(labels_data)[1:]
    data_pad = np.pad(labels_data,100,'constant',constant_values= 0)
    for ID in tqdm(IDset, file=sys.stdout):
        if ID > 20000: continue
        data = data_pad.copy()
        data[data_pad!=ID] = 0
        data[data_pad==ID] = 1
        if np.sum(data) < 200: continue

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
        data_cube = data[center_region[0]-75:center_region[0]+75, center_region[1]-75:center_region[1]+75,center_region[2]-75:center_region[2]+75]

        # 判断是2d/3d
        if data_cube[74,75,75] == 0:
            center, evecs, radii = one_label_2d_fit(data_cube[75])
            v_center = center_region - 100
        else:
            center, evecs, radii = one_label_fit(data_cube)
            # v_center 为原label mrc下的坐标
            v_center = center_region -75 + center - 100
        #label value 为 ID
        info={'name':'vesicle_'+str(ID),'center':v_center.tolist(),'radii':radii.tolist(),'evecs':evecs.tolist(),'CCF':'1'}
        results.append(info)
    #vesicle_info={'pixelsize':str(17.14),'number':str(len(results)),'vesicles':results}
    vesicle_info={'vesicles':results}
    if jsonfile is not None:    
        with open(jsonfile,"w") as out:
            json.dump(vesicle_info,out)
    return vesicle_info

#去重
def update_mrc(data):
    data_mask = data.copy()
    data_mask[data>10000] = 1
    data_mask[data<10000] = 0
    data_10000 = data_mask
    label_10000 = label(data_10000) + 10000
    data_new = np.zeros_like(data_10000)
    data_new = label_10000 * data_mask + data * (1-data_mask)
    return data_new

if __name__ == "__main__":

    import argparse
    import os
    import time
    import mrcfile
    import numpy as np

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--label', type=str, required=True, help='label mrc file')
    parser.add_argument('--jsonfile', type=str, help='output json file')

    args = parser.parse_args()

    label_path = args.label
    if not os.path.exists(label_path):
        raise ValueError(f"Label file {label_path} does not exist.")

    # 设置 jsonfile 默认值为 label 的对应 json 文件
    if args.jsonfile is None:
        args.jsonfile = label_path.replace("_label_vesicle.mrc", "_vesicle.json")

    # 如果已经存在 json 文件，并且备份文件不存在，则进行备份
    backup_jsonfile = args.jsonfile.replace(".json", "_bak.json")
    if os.path.exists(args.jsonfile) and not os.path.exists(backup_jsonfile):
        os.system(f'mv {args.jsonfile} {backup_jsonfile}')

    # 读取并处理 mrc 标签数据
    with mrcfile.open(label_path) as m:
        labeldata = m.data.astype(np.int16)

    # 如果备份的 MRC 文件不存在，则备份原始的 MRC 文件
    backup_mrcfile = label_path.replace(".mrc", "_bak.mrc")
    if not os.path.exists(backup_mrcfile):
        os.system(f'mv {label_path} {backup_mrcfile}')

    # 更新并保存新的 MRC 数据
    label_data = update_mrc(labeldata).astype(np.int16)
    with mrcfile.new(label_path, overwrite=True) as m:
        m.set_data(label_data.astype(np.int16))

    # 生成 json 文件
    t1 = time.time()
    vesicle_info = labels2json(label_data, args.jsonfile)
    print(f'done json generating, cost {time.time() - t1} s')