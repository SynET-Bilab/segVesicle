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


def morph_process(
    mask, area_file, pixelsize=17.14, radius=10
):
    # 1. closing and opening process of vesicle mask. 2. label the vesicles.
    # 3. exclude false vesicles by counting their volumes and thresholding, return only vesicle binary mask
    # 4. extract boundaries and labels them
    # 5. extract labeled individual vesicle boundary, convert into points vectors and output them.

    with mrcfile.open(mask) as f:
        tomo_seg = f.data
    tomo_mask = tomo_seg.copy().astype(np.int8)
    area_mask = boundary_mask(tomo_mask, area_file, pixelsize)
    tomo_mask *= area_mask

    # transform mask into uint8
    bimask = np.round(tomo_mask).astype(np.uint8)
    shape = bimask.shape

    # extract labeled mask whose area more than a threshold
    # (just after prediction, some vesicles will be predicted to be connected)
    area_thre = radius**3
    # bimask = dilation(bimask, cube(2))
    labeled_pre = label(bimask)
    pre_pro = np.zeros(labeled_pre.shape)
    idx_pre = get_indices_sparse(labeled_pre)
    num_pre = np.max(labeled_pre)

    logging.info("\nFirst separatation of the mask by volume thresholding\n")
    for i in tqdm(range(1, num_pre + 1), file=sys.stdout):
        if idx_pre[i][0].shape[0] > area_thre * 15:
            pre_pro[idx_pre[i][0], idx_pre[i][1], idx_pre[i][2]] = 1
            labeled_pre[idx_pre[i][0], idx_pre[i][1], idx_pre[i][2]] = 0
    labeled_pre[labeled_pre > 1] = 1

    kernel_pre = cube(11)
    pre_pro = opening(pre_pro, kernel_pre)
    pre_pro = erosion(pre_pro, cube(3))
    labeled_pre_pro = label(
        pre_pro
    )  # process linked vesicles just after prediction, Part 1

    logging.info("\nFix the broken vesicles\n")
    # for other vesicles
    if True:
        kernel_xy = np.reshape([1 for i in range(9)], (3, 3, 1))
        closing_opening_xy = closing(labeled_pre, kernel_xy)
    else:
        closing_opening_xy = labeled_pre
    if True:
        kernel = np.reshape([1 for i in range(12)], (2, 2, 3))
        closing_opening = closing(closing_opening_xy, kernel)
    else:
        closing_opening = closing_opening_xy

    # label all connected regions
    labeled = label(closing_opening)
    post_pro = np.zeros(labeled.shape)

    idx = get_indices_sparse(labeled)
    num = np.max(labeled)

    logging.info("\nSecond separation of the mask by volume thresholding\n")
    for i in tqdm(range(1, num + 1), file=sys.stdout):
        if idx[i][0].shape[0] < area_thre:
            labeled[idx[i][0], idx[i][1], idx[i][2]] = 0
        elif idx[i][0].shape[0] > area_thre * 12:
            post_pro[idx[i][0], idx[i][1], idx[i][2]] = 1  # record positon of linked vesicles from closing, Part 2
            labeled[idx[i][0], idx[i][1], idx[i][2]] = 0  # main vesicles here, Part 3

    labeled = label(labeled)  # update num of Part3
    num = np.max(labeled)

    # process for Part2
    kernel_p = cube(5)
    post_pro = opening(post_pro, kernel_p)
    # post_pro = erosion(post_pro, cube(3))
    labeled_post_pro = label(post_pro)
    num_post = np.max(labeled_post_pro)

    labeled_post_pro += num
    labeled_post_pro[labeled_post_pro == num] = 0  # update num of Part2

    num += num_post  # update total num of vesicles(except pre_pro)
    labeled_pre_pro += num
    labeled_pre_pro[labeled_pre_pro == num] = 0  # update num of label for part 1
    labeled = labeled + labeled_post_pro + labeled_pre_pro
    num = np.max(labeled)

    # for main vesicles
    filtered = (labeled >= 1).astype(np.uint8)
    logging.info("\ncomplete filtering\n")
    boundaries = filtered - erosion(filtered, cube(3))
    # label the boundaries of vesicles
    bd_labeled = label(boundaries)
    # the number of labeled vesicle
    num = np.max(bd_labeled)
    # vesicle list elements: np.where return point cloud positions whose shape is (3,N)
    idx = get_indices_sparse(bd_labeled)
    vesicle_list = [np.swapaxes(np.array(idx[i]), 0, 1) for i in range(1, num + 1)]

    return vesicle_list, shape


def density_fit(data_iso, center, radius):
    """input center(z,y,x), output center(z,y.x), both in array"""
    shape = data_iso.shape
    # padwidth = int(max(-min(center-radius), -min(np.array(shape)-1-center-radius),0))+5
    padwidth = 10
    maxvalue = np.max(data_iso)
    data_pad = np.pad(data_iso, padwidth, "constant", constant_values=maxvalue)

    center = np.round(center + padwidth).astype(np.int16)
    cube_ = data_pad[
        center[0] - int(radius) - 5 : center[0] + int(radius) + 5 + 1,
        center[1] - int(radius) - 5 : center[1] + int(radius) + 5 + 1,
        center[2] - int(radius) - 5 : center[2] + int(radius) + 5 + 1,
    ]
    cube_ = ndimage.gaussian_filter(cube_, sigma=1)
    cube_reverse = -cube_.astype(np.float32)
    cube_normalize = (cube_reverse - np.min(cube_reverse)) / (
        np.max(cube_reverse) - np.min(cube_reverse)
    )

    mask = ball(cube_.shape[0] // 2)
    mask_circle = cube_.copy()
    p = np.percentile(cube_, 50)
    mask_circle[cube_ < p] = 1
    mask_circle[cube_ >= p] = 0
    mean_circle = np.sum(mask_circle * cube_) / np.sum(mask_circle)

    cube_m = cube_.copy()
    cube_m[cube_ < mean_circle] = 1
    cube_m[cube_ >= mean_circle] = 0

    cube_m_mask = mask * cube_m
    databool = cube_m_mask > 0
    cube_m_mask = remove_small_objects(databool, min_size=50).astype(np.int8)

    open = opening(cube_m_mask)
    databool = open > 0
    opened = remove_small_objects(databool, min_size=50).astype(np.int16)
    l = label(opened, connectivity=1)

    d_min = 99999
    label_vaule = 0
    for i in range(np.max(l)):
        points_i = np.where(l == (i + 1))
        points_z = points_i[0]
        points_y = points_i[1]
        points_x = points_i[2]
        center_i = np.array([np.mean(points_z), np.mean(points_y), np.mean(points_x)])
        center_label = np.array([1, 1, 1]) * l.shape[0] // 2
        d = dis(center_i, center_label)
        if d < d_min and len(points_z) > 200:
            d_min = d
            label_vaule = i + 1
    labeled = np.zeros_like(l)
    labeled[l == label_vaule] = 1
    if d_min == 99999:  # if the num of points to fit is too small (<200)
        return [None, None, None, 0]
    if np.sum(labeled) / np.sum(open) < 0.8:
        labeled = opened
    idx = get_indices_sparse(labeled)
    vesicle_points = np.swapaxes(np.array(idx[1]), 0, 1)
    [center_cube, evecs, radii] = ef.ellipsoid_fit(vesicle_points)
    if np.min(center_cube) < 0:  # if the shape of fitted ellipsoid is too strange
        return [None, None, None, 0]

    tm = template(radii, center_cube, evecs, cube_.shape)
    ccf = CCF(cube_normalize, tm)
    [center_fit, evecs_fit, radii_fit] = [
        center - padwidth + center_cube - cube_.shape[0] // 2,
        evecs,
        radii,
    ]

    return [center_fit, evecs_fit, radii_fit, ccf]


def template(radii, center, evecs, shape, d=3):
    # generate a circle shape template
    ellip = mk.ellipsoid_point(radii, center + np.array([25, 25, 25]), evecs)
    ellip_ = []
    cube_ellip = np.zeros((shape[2] + 50, shape[1] + 50, shape[0] + 50))
    for i in range(len(ellip)):
        if (
            ellip[i][0] < cube_ellip.shape[0]
            and ellip[i][1] < cube_ellip.shape[0]
            and ellip[i][2] < cube_ellip.shape[0]
        ):
            ellip_.append(ellip[i])
    ellip_ = np.array(ellip_)
    if len(ellip_) < 5:
        return cube_ellip
    cube_ellip[ellip_[:, 0], ellip_[:, 1], ellip_[:, 2]] = 1
    cube_ellip = closing(cube_ellip, cube(d))
    circle = dilation(cube_ellip, cube(d)) - erosion(cube_ellip, cube(d))
    tm = ndimage.gaussian_filter(circle, sigma=1).astype(np.float32)
    tm = tm[25:-25, 25:-25, 25:-25]
    
    return tm


def template_2d(radii, center, evecs, shape, d=3):
    # generate a circle shape template
    ellip = mk.ellipsoid_point(radii, center + np.array([25, 25, 25]), evecs)
    ellip_ = []
    cube_ellip = np.zeros((shape[2] + 50, shape[1] + 50, shape[0] + 50))
    for i in range(len(ellip)):
        if (
            ellip[i][0] < cube_ellip.shape[0]
            and ellip[i][1] < cube_ellip.shape[0]
            and ellip[i][2] < cube_ellip.shape[0]
        ):
            ellip_.append(ellip[i])
    ellip_ = np.array(ellip_)
    if len(ellip_) < 5:
        return cube_ellip[cube_ellip.shape[0] // 2]
    cube_ellip[ellip_[:, 0], ellip_[:, 1], ellip_[:, 2]] = 1
    img = cube_ellip[cube_ellip.shape[0] // 2]
    cube_ellip = closing(cube_ellip, cube(d))
    img = closing(img, square(d))
    circle = dilation(img, square(d)) - erosion(img, square(d))
    tm = ndimage.gaussian_filter(circle, sigma=1).astype(np.float32)
    tm = tm[25:-25, 25:-25]
    
    return tm


def CCF(img, template):
    """ """
    img_mean = np.mean(img)
    tm_mean = np.mean(template)
    if np.sum((template - tm_mean) ** 2) < 0.0001:
        return 0
    else:
        ccf = np.sum((img - img_mean) * (template - tm_mean)) / np.sqrt(
            np.sum((img - img_mean) ** 2) * np.sum((template - tm_mean) ** 2)
        )
    return ccf


def generate_2d_gaussian_weights(size, sigma=1.0):
    """
    生成2维高斯权重矩阵，中心值最大，周围衰减。
    - size: 数组的尺寸（假设为立方体，即 size x size ）
    - sigma: 控制衰减速度，sigma越小中心越突出
    """
    center = size // 2
    x, y = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    x_centered = x - center
    y_centered = y - center
    squared_dist = x_centered**2 + y_centered**2
    gauss = np.exp(-squared_dist / (2 * sigma**2))
    # 可选：归一化到0-1范围，或保持中心为1
    gauss /= gauss.max()  # 使中心权重为1，周围按比例衰减
    return gauss


def density_fit_2d(data_iso, center, radius):
    """input center(z,y,x), output center(z,y.x), both in array"""

    shape = data_iso.shape
    # padwidth = int(max(-min(center-radius), -min(np.array(shape)-1-center-radius),0))+5
    padwidth = 10
    maxvalue = np.max(data_iso)
    data_pad = np.pad(data_iso, padwidth, "constant", constant_values=maxvalue)

    center = np.round(center + padwidth).astype(np.int16)
    cube_ = data_pad[
        center[0] - int(radius) - 5 : center[0] + int(radius) + 5 + 1,
        center[1] - int(radius) - 5 : center[1] + int(radius) + 5 + 1,
        center[2] - int(radius) - 5 : center[2] + int(radius) + 5 + 1,
    ]
    # cube_2=data_pad[center[0]-int(radius)-15: center[0]+int(radius)+15+1,center[1]-int(radius)-15: center[1]+int(radius)+15+1,center[2]-int(radius)-15: center[2]+int(radius)+15+1]
    # with mrcfile.new('/home/lvzy/test/ves_seg/cube.mrc',overwrite=True) as m:
    #     m.set_data(cube_2)
    img = cube_[cube_.shape[0] // 2, :, :]
    img = ndimage.gaussian_filter(img, sigma=1)

    img_reverse = -img.astype(np.float32)
    img_normalize = (img_reverse - np.min(img_reverse)) / (
        np.max(img_reverse) - np.min(img_reverse)
    )

    sigma = int(radius) + 5  # 调整sigma控制衰减范围
    gaussian_weights = generate_2d_gaussian_weights(img.shape[0], sigma)
    img_normalize = img_normalize * gaussian_weights  # 直接相乘增强中心
    # with mrcfile.new('/home/lvzy/test/ves_seg/img.mrc',overwrite=True) as m:
    #         m.set_data(-img_normalize.astype(np.float32))
    mask = disk(cube_.shape[1] // 2)
    img = -img_normalize
    mask_circle = img.copy()
    p = np.percentile(img, 50)
    mask_circle[img < p] = 1
    mask_circle[img >= p] = 0
    mean_circle = np.sum(mask_circle * img) / np.sum(mask_circle)

    img_m = img.copy()
    img_m[img < mean_circle] = 1
    img_m[img >= mean_circle] = 0
    # img_m=img.copy()
    # avg = 0.5 * (np.min(img)+np.max(img))
    # img_m[img<avg]=1
    # img_m[img>=avg]=0

    img_m_mask = mask * img_m
    open = opening(img_m_mask, square(2))
    databool = open > 0
    open = remove_small_objects(databool, min_size=10).astype(np.int16)
    # open2=np.pad(open,10,'constant',constant_values= 0)
    # with mrcfile.new('/home/lvzy/test/ves_seg/open.mrc',overwrite=True) as m:
    #     m.set_data(open2)
    l = label(open, connectivity=1)
    labeled = open
    for i in range(np.max(l)):
        points_i = np.where(l == (i + 1))
        points_y = points_i[0]
        points_x = points_i[1]
        center_i = np.array([np.mean(points_y), np.mean(points_x)])
        center_label = np.array([1, 1]) * l.shape[0] // 2
        d = dis(center_i, center_label)
        if d > radius and len(points_y) > 10:
            labeled[l == i + 1] = 0

    cube_m_mask = np.zeros_like(cube_)
    cube_m_mask[cube_.shape[0] // 2] = labeled
    # cube_m_mask2=np.pad(cube_m_mask,10,'constant',constant_values= 0)
    # with mrcfile.new('/home/lvzy/test/ves_seg/pts.mrc',overwrite=True) as m:
    #     m.set_data(cube_m_mask2)
    cloud = np.where(cube_m_mask > 0)
    x = np.asarray(cloud[2])
    y = np.asarray(cloud[1])
    z = np.asarray(cloud[0])[0]
    [center_cube, evecs, radii] = ef.ellipse_fit_n(x, y, z)
    if np.min(center_cube) < 0:  # if the shape of fitted ellipsoid is too strange
        return [None, None, None, 0]

    tm = template_2d(radii, center_cube, evecs, cube_.shape)
    ccf = CCF(img_normalize, tm)
    [center_fit, evecs_fit, radii_fit] = [
        center - padwidth + center_cube - cube_.shape[0] // 2,
        evecs,
        radii,
    ]

    return [center_fit, evecs_fit, radii_fit, ccf]


def fit_6pts(data_iso, points):
    x = []
    y = []
    z = points[0][0]
    for i in range(len(points)):
        x.append(points[i][2])
        y.append(points[i][1])
    [center_cube, evecs, radii] = ef.ellipse_fit(x, y, z)
    ccf = 1
    # radius = np.max(radii).astype(np.int8)
    # cube_shape=(2*radius + 50,2*radius + 50,2*radius + 50)
    # tm = template_2d(radii, center_cube, evecs, cube_shape)

    # shape = data_iso.shape
    # #padwidth = int(max(-min(center_cube-radius), -min(np.array(shape)-1-center_cube-radius),0))+5
    # padwidth = 20
    # maxvalue=np.max(data_iso)
    # data_pad = np.pad(data_iso,padwidth,'constant',constant_values= maxvalue)
    # center = np.round(center_cube+padwidth).astype(np.int16)
    # cube_=data_pad[center[0]-int(radius)-5: center[0]+int(radius)+5+1,center[1]-int(radius)-5: center[1]+int(radius)+5+1,center[2]-int(radius)-5: center[2]+int(radius)+5+1]
    # img = cube_[cube_.shape[0]//2,:,:]
    # img = ndimage.gaussian_filter(img,sigma=1)
    # img_reverse = -img
    # img_normalize = (img_reverse - np.min(img_reverse))/(np.max(img_reverse)-np.min(img_reverse))
    # ccf = CCF(img_normalize,tm)
    return [center_cube, evecs, radii, ccf]


def measure_one(idx, data, vesicle_list, min_radius):

    [center0, evecs, radii] = ef.ellipsoid_fit(vesicle_list[idx])
    if (
        min(center0 - max(radii)) <= 0
        or min(np.array(data.shape) - 1 - center0 - max(radii)) <= 0
    ):
        return

    def if_normal(radii, threshold=0.22):
        if np.std(radii) / np.mean(radii) > threshold:
            a = False
        elif np.mean(radii) < 0.6 * min_radius or np.mean(radii) > min_radius * 4:
            a = False
        else:
            a = True
        return a

    [center, evecs, radii, ccf] = density_fit(data, center0, np.max(radii))
    if ccf < 0.3:  # delete wrong segments
        return

    if if_normal(radii):
        info = {
            "name": "vesicle_" + str(idx),
            "center": center.tolist(),
            "radii": radii.tolist(),
            "evecs": evecs.tolist(),
            "CCF": str(ccf),
        }
        return info


def vesicle_measure(data, vesicle_list, shape, min_radius, outfile):
    """ """
    results = []
    global in_count
    global sup_in_count
    in_count = 0
    sup_in_count = 0

    logging.info("\nStart vesicle measurement\n")

    idxs = range(len(vesicle_list))
    for i in tqdm(range(len(vesicle_list)), file=sys.stdout):
        continue
    # pool = multiprocessing.Pool(multiprocessing.cpu_count())
    poolnum = min(multiprocessing.cpu_count(), 4)
    pool = multiprocessing.Pool(poolnum)
    results = pool.starmap(
        measure_one, [(i, data, vesicle_list, min_radius) for i in idxs]
    )
    results = list(filter(None, results))

    # return vesicle information dict and save as json
    vesicle_info = {"vesicles": results}

    if outfile is not None:
        with open(outfile, "w") as out:
            json.dump(vesicle_info, out)
    
    return vesicle_info


def vesicle_rendering(vesicle_file, tomo_dims):
    """ """
    # vesicle file can be json or a info list
    if type(vesicle_file) is str:
        with open(vesicle_file) as f:
            ves = json.load(f)
        vesicle_info = ves["vesicles"]
    else:
        vesicle_info = vesicle_file
    vesicle_tomo = np.zeros(
        np.array(tomo_dims) + np.array([30, 30, 30]), dtype=np.int16
    )
    # vesicle_tomo = np.zeros(np.array(tomo_dims),dtype=np.uint8)
    logging.info("\nrendering vesicle\n")
    # for i,vesicle in enumerate(vesicle_info):
    for i in tqdm(range(len(vesicle_info)), file=sys.stdout):
        ellip_i = mk.ellipsoid_point(
            vesicle_info[i]["radii"],
            vesicle_info[i]["center"],
            vesicle_info[i]["evecs"],
        )
        # ellip_i is an array (N,3) of points of a filled ellipsoid
        Id = int(vesicle_info[i]["name"][8:])
        vesicle_tomo[ellip_i[:, 0], ellip_i[:, 1], ellip_i[:, 2]] = Id
        # vesicle_tomo[ellip_i[:,0],ellip_i[:,1],ellip_i[:,2]] = i + 1
        xmin, xmax = np.min(ellip_i[:, 2]), np.max(ellip_i[:, 2])
        ymin, ymax = np.min(ellip_i[:, 1]), np.max(ellip_i[:, 1])
        zmin, zmax = np.min(ellip_i[:, 0]), np.max(ellip_i[:, 0])
        cube_i = vesicle_tomo[zmin : zmax + 1, ymin : ymax + 1, xmin : xmax + 1]
        cube_i = closing(cube_i, cube(3))
        vesicle_tomo[zmin : zmax + 1, ymin : ymax + 1, xmin : xmax + 1] = cube_i

    # vesicle_tomo = closing(vesicle_tomo,cube(3))
    logging.info("{} vesicles in total".format(len(vesicle_info)))
    return vesicle_tomo[0 : tomo_dims[0], 0 : tomo_dims[1], 0 : tomo_dims[2]]


def dis(m, n):
    d = np.linalg.norm(m - n)
    return d


def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size))


def get_indices_sparse(data):
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]


if __name__ == "__main__":

    import argparse
    import time

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--tomo", type=str, default=None, help="tomo file")
    parser.add_argument(
        "--tomo_file", type=str, default=None, help="the isonet_corrected tomo file"
    )
    parser.add_argument(
        "--mask_file", type=str, default=None, help="the output vesicle mask file name"
    )
    parser.add_argument(
        "--label", type=str, default=None, help="draw fitted vesicles as labels"
    )
    parser.add_argument(
        "--min_radius", type=int, default=8, help="minimal radius of targeting vesicles"
    )
    parser.add_argument(
        "--pixelsize",
        type=float,
        default=17.14,
        help="pixelsize(in Angstrom) of original tomo consistent with area file",
    )
    parser.add_argument(
        "--area_file",
        type=str,
        default=None,
        help=".point or .mod file which defines interested area",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="output vesicles file name (xxx.json)",
    )

    args = parser.parse_args()

    # set some default files
    if args.tomo_file is None:
        args.tomo_file = args.tomo + "_wbp_corrected.mrc"
    if args.mask_file is None:
        args.mask_file = args.tomo + "_segment.mrc"
    if args.label is None:
        args.label = args.tomo + "_label_vesicle.mrc"
    if args.output_file is None:
        args.output_file = args.tomo + "_vesicle.json"

    # t1 = time.time()
    # save raw vesicle mask
    with mrcfile.open(args.mask_file) as m:
        bimask = m.data
    shape = bimask.shape
    print("begin morph process")
    vesicle_list, shape = morph_process(
        args.mask_file, args.area_file, pixelsize=args.pixelsize, radius=args.min_radius
    )
    print("done morph process")

    with mrcfile.open(args.tomo_file) as m:
        data_iso = m.data
    vesicle_info = vesicle_measure(
        data_iso, vesicle_list, shape, args.min_radius, args.output_file
    )
    print("done vesicle measuring")

    if args.label is not None:
        ves_tomo = vesicle_rendering(args.output_file, shape)
        # labels = label(ves_tomo).astype(np.float32)
        with mrcfile.new(args.label, overwrite=True) as n:
            n.set_data(ves_tomo.astype(np.int16))

    # print(f"morph cost {time.time()-t1} s")
