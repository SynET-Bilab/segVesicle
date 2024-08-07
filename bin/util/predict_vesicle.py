import time
import logging
import sys
import os
import numpy as np
from tqdm import tqdm
from scipy import ndimage

from skimage.morphology import opening, closing, erosion, dilation, remove_small_objects
from skimage.morphology import cube, ball
from scipy.sparse import csr_matrix
from skimage.measure import label

from segVesicle.bin.boundary_mask import boundary_mask
from segVesicle.models import resunet3D as models
from segVesicle.bin.ellipsoid import ellipsoid_fit as ef
from segVesicle.utils import make_ellipsoid as mk

from util.model_exists import ensure_model_exists


def predict_label(deconv_data, corrected_data):
    model_1 = 'vesicle_seg_model_1.h5'
    model_2 = 'vesicle_seg_model_2.h5'
    
    path_weights1 = '/home/liushuo/Documents/data/model/vesicle_seg_model_1.h5'
    # path_weights1 = ensure_model_exists(model_1)
    path_weights2 = ensure_model_exists(model_2)

    seg1 = segment(path_weights1, corrected_data)
    seg2 = segment(path_weights2, deconv_data)
    labelmap = np.sign(seg1 + seg2).astype(np.int8)
    
    return labelmap
    
def segment(path_weights, data, patch_size=192):
    pcrop = 48  # how many pixels to crop from border

    P = patch_size
    Ncl = 2
    # build network
    net = models.my_model(patch_size, Ncl)
    net.load_weights(path_weights)

    percentile_99_5 = np.percentile(data, 99.5)
    percentile_00_5 = np.percentile(data, 00.5)
    data = np.clip(data, percentile_00_5, percentile_99_5)

    data = (data - np.mean(data)) / np.std(data)  # normalize
    data = np.pad(data, pcrop, mode='constant', constant_values=0)  # 0pad
    dim = data.shape
    l = int(P / 2)
    lcrop = int(l - pcrop)
    step = int(2 * l - 2 * pcrop)
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
                print(f'Segmenting patch {patchCount} / {Npatch} ...')
                patch = data[z - l:z + l, y - l:y + l, x - l:x + l]
                patch = np.reshape(patch, (1, P, P, P, 1))  # reshape for keras [batch,x,y,z,channel]
                pred = net.predict(patch, batch_size=10)
                predArray[z-lcrop:z+lcrop, y-lcrop:y+lcrop, x-lcrop:x+lcrop, :] += np.float16(pred[0, l - lcrop:l + lcrop, l - lcrop:l + lcrop, l - lcrop:l + lcrop, :])
                normArray[z-lcrop:z+lcrop, y-lcrop:y+lcrop, x-lcrop:x+lcrop] += np.ones((P-2*pcrop, P-2*pcrop, P-2*pcrop), dtype=np.int8)
                patchCount += 1
    normArray[normArray == 0] = 1

    # Normalize overlapping regions:
    for C in range(0, Ncl):
        predArray[:, :, :, C] = predArray[:, :, :, C] / normArray
    end = time.time()
    print(f"Model took {int(end - start)} seconds to predict")
    predArray = predArray[pcrop:-pcrop, pcrop:-pcrop, pcrop:-pcrop, :]  # unpad

    labelmap = np.int8(np.argmax(predArray, 3))
    return labelmap
    
def morph_process(labelmap, area_file, pixelsize=17.14, radius=10):
    # 1. closing and opening process of vesicle mask. 2. label the vesicles.
    # 3. exclude false vesicles by counting their volumes and thresholding, return only vesicle binary mask
    # 4. extract boundaries and labels them
    # 5. extract labeled individual vesicle boundary, convert into points vectors and output them.
    tomo_mask = labelmap.astype(np.int8)
    area_mask = boundary_mask(tomo_mask, area_file, pixelsize)
    tomo_mask *= area_mask
    
    # transform mask into uint8
    bimask = np.round(tomo_mask).astype(np.uint8)
    shape = bimask.shape

    # extract labeled mask whose area more than a threshold 
    # (just after prediction, some vesicles will be predicted to be connected)
    area_thre = radius**3
    labeled_pre = label(bimask)
    sup_pro = np.zeros(labeled_pre.shape)
    pre_pro = np.zeros(labeled_pre.shape)
    idx_pre = get_indices_sparse(labeled_pre)
    num_pre = np.max(labeled_pre)

    print('\nFirst separation of the mask by volume thresholding\n')
    for i in tqdm(range(1, num_pre+1)):
        if idx_pre[i][0].shape[0] > area_thre*15:
            pre_pro[idx_pre[i][0], idx_pre[i][1], idx_pre[i][2]] = 1
            labeled_pre[idx_pre[i][0], idx_pre[i][1], idx_pre[i][2]] = 0
    labeled_pre[labeled_pre > 1] = 1

    kernel_pre = cube(11)
    pre_pro = opening(pre_pro, kernel_pre)
    pre_pro = erosion(pre_pro, cube(2))
    labeled_pre_pro = label(pre_pro) #process linked vesicles just after prediction, Part 1

    print('\nFix the broken vesicles\n')
    kernel_xy = np.reshape([1 for i in range(9)], (3, 3, 1))
    closing_opening_xy = opening(labeled_pre, kernel_xy)
    kernel = np.reshape([1 for i in range(12)], (2, 2, 3))
    closing_opening = opening(closing_opening_xy, kernel)

    labeled = label(closing_opening)
    post_pro = np.zeros(labeled.shape)
    idx = get_indices_sparse(labeled)
    num = np.max(labeled)

    print('\nSecond separation of the mask by volume thresholding\n')
    for i in tqdm(range(1, num+1)):
        if idx[i][0].shape[0] < area_thre:
            labeled[idx[i][0], idx[i][1], idx[i][2]] = 0
            if idx[i][0].shape[0] > 0.2*area_thre:
                sup_pro[idx[i][0], idx[i][1], idx[i][2]] = 1
        elif idx[i][0].shape[0] <= 2*area_thre and idx[i][0].shape[0] > area_thre:
            sup_pro[idx[i][0], idx[i][1], idx[i][2]] = 1
        elif idx[i][0].shape[0] > area_thre*12:
            post_pro[idx[i][0], idx[i][1], idx[i][2]] = 1
            labeled[idx[i][0], idx[i][1], idx[i][2]] = 0

    labeled = label(labeled)
    num = np.max(labeled)

    kernel_p = cube(5)
    post_pro = opening(post_pro, kernel_p)
    labeled_post_pro = label(post_pro)
    num_post = np.max(labeled_post_pro)

    labeled_post_pro += num
    labeled_post_pro[labeled_post_pro == num] = 0

    num += num_post
    labeled_pre_pro += num
    labeled_pre_pro[labeled_pre_pro == num] = 0
    labeled = labeled + labeled_post_pro + labeled_pre_pro
    num = np.max(labeled)

    filtered = (labeled >= 1).astype(np.uint8)
    print('\ncomplete filtering\n')
    boundaries = filtered - erosion(filtered, cube(3))
    bd_labeled = label(boundaries)
    idx = get_indices_sparse(bd_labeled)
    vesicle_list = [np.swapaxes(np.array(idx[i]), 0, 1) for i in range(1, np.max(bd_labeled)+1)]
    
    return vesicle_list, shape
    

def vesicle_measure(data, vesicle_list, shape, min_radius):
    results = []
    global in_count
    global sup_in_count
    in_count = 0
    sup_in_count = 0

    def if_normal(radii, threshold=0.22):
        if np.std(radii) / np.mean(radii) > threshold:
            return False
        elif np.mean(radii) < 0.6 * min_radius or np.mean(radii) > min_radius * 4:
            return False
        else:
            return True

    logging.info('\nStart vesicle measurement\n')
    for i in tqdm(range(len(vesicle_list)), file=sys.stdout):
        [center0, evecs, radii] = ef.ellipsoid_fit(vesicle_list[i])
        if min(center0 - max(radii)) <= 0 or min(np.array(data.shape) - 1 - center0 - max(radii)) <= 0:
            continue

        [center, evecs, radii, ccf] = density_fit(data, center0, np.max(radii))
        if ccf < 0.3:
            continue

        if if_normal(radii):
            info = {
                'name': 'vesicle_' + str(i),
                'center': center.tolist(),
                'radii': radii.tolist(),
                'evecs': evecs.tolist(),
                'CCF': str(ccf)
            }
            results.append(info)

    vesicle_info = {'vesicles': results}
    return vesicle_info

def density_fit(data_iso,center,radius):
    '''input center(z,y,x), output center(z,y.x), both in array
    '''
    shape = data_iso.shape
    padwidth = int(max(-min(center-radius), -min(np.array(shape)-1-center-radius),0))+5
    maxvalue=np.max(data_iso)
    data_pad = np.pad(data_iso,padwidth,'constant',constant_values= maxvalue)


    center = np.round(center+padwidth).astype(np.int16)
    cube_=data_pad[center[0]-int(radius)-5: center[0]+int(radius)+5+1,center[1]-int(radius)-5: center[1]+int(radius)+5+1,center[2]-int(radius)-5: center[2]+int(radius)+5+1]
    cube_ = ndimage.gaussian_filter(cube_,sigma=1)
    cube_reverse = -cube_
    cube_normalize = (cube_reverse - np.min(cube_reverse))/(np.max(cube_reverse)-np.min(cube_reverse))

    mask=ball(cube_.shape[0]//2)
    mask_circle=cube_.copy()
    p=np.percentile(cube_, 50)
    mask_circle[cube_<p]=1
    mask_circle[cube_>=p]=0
    mean_circle=np.sum(mask_circle*cube_)/np.sum(mask_circle)

    cube_m=cube_.copy()
    cube_m[cube_<mean_circle]=1
    cube_m[cube_>=mean_circle]=0
    cube_m_mask=mask*cube_m
    databool=cube_m_mask >0
    cube_m_mask=remove_small_objects(databool, min_size=50).astype(np.int8)
    
    open=opening(cube_m_mask)
    databool=open >0
    opened=remove_small_objects(databool, min_size=50).astype(np.int16)
    if np.sum(opened) < 1000:
        return [None, None, None, 0]
    #erded=erosion(opened,cube(2))
    idx=get_indices_sparse(opened)
    
    vesicle_points=np.swapaxes(np.array(idx[1]),0,1)
    [center_cube, evecs, radii]=ef.ellipsoid_fit(vesicle_points)


    tm = template(radii, center_cube, evecs, cube_.shape)
    ccf = CCF(cube_normalize,tm)
    [center_fit, evecs_fit, radii_fit]=[center-padwidth+center_cube-cube_.shape[0]//2, evecs, radii]

    return [center_fit, evecs_fit, radii_fit, ccf]

def get_indices_sparse(data):
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]


def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)), shape=(data.max()+1, data.size))

def template(radii, center, evecs, shape, d=3):
    #generate a circle shape template
    ellip = mk.ellipsoid_point(radii, center+np.array([25,25,25]), evecs)
    cube_ellip = np.zeros((shape[2]+50,shape[1]+50,shape[0]+50))


    if cube_ellip.shape[0] <= np.max(ellip):
        tm = 1-cube_ellip
    else:
        cube_ellip[ellip[:,0],ellip[:,1],ellip[:,2]] = 1
        cube_ellip=closing(cube_ellip,cube(d))
        circle = dilation(cube_ellip,cube(d)) - erosion(cube_ellip,cube(d))
        tm = ndimage.gaussian_filter(circle,sigma=1).astype(np.float32)
    tm = tm[25:-25,25:-25,25:-25]
    return tm

def CCF(img,template):
    '''
    '''
    img_mean = np.mean(img)
    tm_mean = np.mean(template)
    if np.sum((template - tm_mean)**2)<0.0001:
        return 0
    else:
        ccf = np.sum((img - img_mean) * (template - tm_mean)) / np.sqrt(np.sum((img - img_mean)**2) * np.sum((template - tm_mean)**2))
    return ccf

def vesicle_rendering(vesicle_info, tomo_dims):
    vesicle_info_list = vesicle_info['vesicles']
    vesicle_tomo = np.zeros(np.array(tomo_dims) + np.array([30, 30, 30]), dtype=np.int16)
    logging.info('\nrendering vesicle\n')
    for i in tqdm(range(len(vesicle_info_list)), file=sys.stdout):
        ellip_i = mk.ellipsoid_point(vesicle_info_list[i]['radii'], vesicle_info_list[i]['center'], vesicle_info_list[i]['evecs'])
        vesicle_tomo[ellip_i[:, 0], ellip_i[:, 1], ellip_i[:, 2]] = i + 1
        xmin, xmax = np.min(ellip_i[:, 2]), np.max(ellip_i[:, 2])
        ymin, ymax = np.min(ellip_i[:, 1]), np.max(ellip_i[:, 1])
        zmin, zmax = np.min(ellip_i[:, 0]), np.max(ellip_i[:, 0])
        cube_i = vesicle_tomo[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]
        cube_i = closing(cube_i, cube(3))
        vesicle_tomo[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1] = cube_i

    logging.info('{} vesicles in total'.format(len(vesicle_info_list)))
    return vesicle_tomo[0:tomo_dims[0], 0:tomo_dims[1], 0:tomo_dims[2]]