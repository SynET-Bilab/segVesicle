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

from segVesicle.bin.morph_bak import get_indices_sparse
from segVesicle.bin.boundary_mask import boundary_mask
from segVesicle.utils import make_ellipsoid as mk
from segVesicle.bin.ellipsoid import ellipsoid_fit as ef

def mw2d(dim,missingAngle=(30,30)):
    mw=np.zeros((dim,dim),dtype='float32')
    missingAngle = np.array(missingAngle,dtype='float32')
    missing=np.pi/180*(90-missingAngle)
    for i in range(dim):
        for j in range(dim):
            y=(i-dim/2)
            x=(j-dim/2)
            if x==0:# and y!=0:
                theta=np.pi/2

            else:
                theta=abs(np.arctan(y/x))

            if x**2+y**2<=min(dim/2,dim/2)**2:
                if x > 0 and y > 0 and theta < missing[0]:
                    mw[i,j]=1#np.cos(theta)
                if x < 0 and y < 0 and theta < missing[0]:
                    mw[i,j]=1#np.cos(theta)
                if x > 0 and y < 0 and theta < missing[1]:
                    mw[i,j]=1#np.cos(theta)
                if x < 0 and y > 0 and theta < missing[1]:
                    mw[i,j]=1#np.cos(theta)
            if int(y) == 0:
                mw[i,j]=1
    return mw

def apply_wedge(ori_data, ld1 = 1, ld2 =0, mw3d = None):
    if mw3d == None:
        data = np.rot90(ori_data, k=1, axes=(0,1)) #clock wise of counter clockwise??
        mw = mw2d(data.shape[1])
        #if inverse:
        #    mw = 1-mw
        mw = mw * ld1 + (1-mw) * ld2
        outData = np.zeros(data.shape,dtype='float32')
        mw_shifted = np.fft.fftshift(mw)
        for i, item in enumerate(data):
            outData_i=np.fft.ifft2(mw_shifted * np.fft.fft2(item))
            outData[i] = np.real(outData_i)

        outData.astype(np.float32)
        outData=np.rot90(outData, k=3, axes=(0,1))
        return outData
    else:
        import mrcfile
        with mrcfile.open(mw3d, 'r') as mrc:
            mw = mrc.data
        mw = np.fft.fftshift(mw)
        mw = mw * ld1 + (1-mw) * ld2

        f_data = np.fft.fftn(ori_data)
        outData = mw*f_data
        inv = np.fft.ifftn(outData)
        outData = np.real(inv).astype(np.float32)
    return outData

def one_label_fit(data):
    
    idx=get_indices_sparse(data)
    vesicle_points=np.swapaxes(np.array(idx[1]),0,1)
    [center, evecs, radii]=ef.ellipsoid_fit(vesicle_points)
    return center, evecs, radii

def v_mw_compare(labelfile,jsonfile):



    with mrcfile.open(labelfile) as m:
        la_ori=m.data

    la = la_ori.copy()
    la_pad = np.pad(la,30,'constant',constant_values= 0)
    la_pad = la_pad.astype(np.int16)
    regions = regionprops(la_pad)
    results=[]
    for rgn in tqdm(regions, file=sys.stdout):
        v_ID = rgn.label
        v_center = rgn.centroid #zyx
        v_center = np.round(v_center).astype(np.int16)
        v_one = np.zeros_like(la_pad)
        v_one[la_pad==v_ID] = 1
        v_cube = v_one[v_center[0]-25:v_center[0]+25,v_center[1]-25:v_center[1]+25,v_center[2]-25:v_center[2]+25]
        # 判断是2d/3d
        if v_cube[24,25,25] == 0:
            continue

        v_shell = dilation(v_cube,ball(1)) - erosion(v_cube,ball(1))
        v_shell_mw_ori = apply_wedge(v_shell)
        p=np.percentile(v_shell_mw_ori, 97.5)
        v_shell_mw=v_shell_mw_ori.copy()
        v_shell_mw[v_shell_mw_ori<p]=0
        v_shell_mw[v_shell_mw_ori>p]=1
        v_shell_mw = v_shell_mw.astype(np.int8)
        center, evecs, radii = one_label_fit(v_shell)
        v_c = v_center - 25 + center -30
        center_mw, evecs_mw, radii_mw = one_label_fit(v_shell_mw)
        v_c_mw = v_center - 25 + center_mw -30
        info_ori = {'center':v_c.tolist(),'radii':radii.tolist(),'evecs':evecs.tolist()}
        info_mw = {'center':v_c_mw.tolist(),'radii':radii_mw.tolist(),'evecs':evecs_mw.tolist()}
        info={'name':'vesicle_'+str(v_ID),'ori':info_ori, 'mw':info_mw}
        results.append(info)
    vesicle_info={'vesicles':results}
    with open(jsonfile,"w") as out:
        json.dump(vesicle_info,out)



if __name__ == "__main__":

    import argparse
    import os
    import time
    import mrcfile
    import numpy as np

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-L','--label', type=str, required=True, help='label mrc file')
    parser.add_argument('-J','--jsonfile', type=str, help='output json file')

    args = parser.parse_args()

    label_path = args.label
    if not os.path.exists(label_path):
        raise ValueError(f"Label file {label_path} does not exist.")

    # 设置 jsonfile 默认值为 label 的对应 json 文件
    if args.jsonfile is None:
        args.jsonfile = label_path.replace("_label_vesicle.mrc", "_mw_compare.json")

    # 如果已经存在 json 文件，并且备份文件不存在，则进行备份
    backup_jsonfile = args.jsonfile.replace(".json", "_bak.json")
    if os.path.exists(args.jsonfile) and not os.path.exists(backup_jsonfile):
        os.system(f'mv {args.jsonfile} {backup_jsonfile}')



    # 生成 json 文件
    t1 = time.time()
    vesicle_info = v_mw_compare(label_path, args.jsonfile)
    print(f'done json generating, cost {time.time() - t1} s')

    