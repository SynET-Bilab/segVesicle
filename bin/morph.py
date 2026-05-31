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
from skimage.morphology import (
    binary_closing,
    binary_erosion,
    binary_opening,
    opening,
    closing,
    erosion,
    dilation,
    remove_small_objects,
)
from skimage.morphology import cube, ball, disk, square

from segVesicle.bin.boundary_mask import boundary_mask
from segVesicle.utils import make_ellipsoid as mk
from segVesicle.bin.ellipsoid import ellipsoid_fit as ef


_MEASURE_DATA = None
_MEASURE_VESICLE_LIST = None
_MEASURE_MIN_RADIUS = None
_MEASURE_DATA_MAX = None


def _label_mask(labeled, label_ids, label_count=None):
    if len(label_ids) == 0:
        return np.zeros(labeled.shape, dtype=bool)
    if label_count is None:
        label_count = int(labeled.max()) + 1
    lookup = np.zeros(label_count, dtype=bool)
    lookup[np.asarray(label_ids, dtype=np.int64)] = True
    return lookup[labeled]


def _foreground_label_ids(sizes, condition):
    ids = np.flatnonzero(condition)
    return ids[ids != 0]


def _points_by_label_bboxes(labeled):
    vesicle_list = []
    for label_id, slices in enumerate(ndimage.find_objects(labeled), start=1):
        if slices is None:
            continue
        local = np.nonzero(labeled[slices] == label_id)
        if local[0].size == 0:
            continue
        offsets = [sl.start for sl in slices]
        points = np.column_stack(
            (
                local[0] + offsets[0],
                local[1] + offsets[1],
                local[2] + offsets[2],
            )
        ).astype(np.int64, copy=False)
        vesicle_list.append(points)
    return vesicle_list


def _extract_padded_cube(data, center, radius, pad_value):
    half_width = int(radius) + 5
    center_i = np.round(center).astype(np.int64)
    start = center_i - half_width
    stop = center_i + half_width + 1
    cube_shape = tuple((stop - start).astype(np.int64))
    cube_data = np.full(cube_shape, pad_value, dtype=data.dtype)

    data_shape = np.asarray(data.shape)
    src_start = np.maximum(start, 0)
    src_stop = np.minimum(stop, data_shape)
    if np.any(src_start >= src_stop):
        return cube_data, center_i

    dst_start = src_start - start
    dst_stop = dst_start + (src_stop - src_start)
    src_slices = tuple(slice(int(s), int(e)) for s, e in zip(src_start, src_stop))
    dst_slices = tuple(slice(int(s), int(e)) for s, e in zip(dst_start, dst_stop))
    cube_data[dst_slices] = data[src_slices]
    return cube_data, center_i


def _init_measure_worker(data, vesicle_list, min_radius, data_max):
    global _MEASURE_DATA
    global _MEASURE_VESICLE_LIST
    global _MEASURE_MIN_RADIUS
    global _MEASURE_DATA_MAX
    _MEASURE_DATA = data
    _MEASURE_VESICLE_LIST = vesicle_list
    _MEASURE_MIN_RADIUS = min_radius
    _MEASURE_DATA_MAX = data_max


def morph_process(mask, area_file, pixelsize=17.14, radius=10, value_check=True):
    """
    mask MUST be binary. This is correct for segment.py/predict_label output.
    For labeled vesicle maps, use label2json.py instead.
    """

    if isinstance(mask, np.ndarray):
        tomo_seg = mask
    else:
        with mrcfile.open(mask) as f:
            tomo_seg = f.data

    if value_check and (np.min(tomo_seg) < 0 or np.max(tomo_seg) > 1):
        raise ValueError(
            "morph_process expects a binary segmentation mask with values in [0, 1]. "
            "For labeled vesicle maps, use label2json.py."
        )

    tomo_mask = tomo_seg.copy().astype(np.int8)
    area_mask = boundary_mask(tomo_mask, area_file, pixelsize)
    tomo_mask *= area_mask

    bimask = np.round(tomo_mask).astype(bool)
    shape = bimask.shape

    # extract labeled mask whose area more than a threshold
    # (just after prediction, some vesicles will be predicted to be connected)
    area_thre = radius**3
    # bimask = dilation(bimask, cube(2))
    labeled_pre = label(bimask)
    sizes_pre = np.bincount(labeled_pre.ravel())
    pre_ids = _foreground_label_ids(sizes_pre, sizes_pre > area_thre * 15)
    pre_pro = _label_mask(labeled_pre, pre_ids, len(sizes_pre))

    logging.info("\nFirst separatation of the mask by volume thresholding\n")
    labeled_pre[pre_pro] = 0
    labeled_pre = labeled_pre > 0

    kernel_pre = cube(11, decomposition="sequence")
    pre_pro = binary_opening(pre_pro, footprint=kernel_pre)
    pre_pro = binary_erosion(pre_pro, footprint=cube(3, decomposition="sequence"))
    labeled_pre_pro = label(pre_pro)  # process linked vesicles just after prediction, Part 1

    logging.info("\nFix the broken vesicles\n")
    # for other vesicles
    kernel_xy = np.ones((3, 3, 1), dtype=bool)
    closing_opening_xy = binary_closing(labeled_pre, footprint=kernel_xy)
    kernel = np.ones((2, 2, 3), dtype=bool)
    closing_opening = binary_closing(closing_opening_xy, footprint=kernel)

    # label all connected regions
    labeled = label(closing_opening)
    sizes = np.bincount(labeled.ravel())
    small_ids = _foreground_label_ids(sizes, sizes < area_thre)
    post_ids = _foreground_label_ids(sizes, sizes > area_thre * 12)
    post_pro = _label_mask(labeled, post_ids, len(sizes))
    remove_ids = np.concatenate((small_ids, post_ids))
    remove_mask = _label_mask(labeled, remove_ids, len(sizes))

    logging.info("\nSecond separation of the mask by volume thresholding\n")
    labeled[remove_mask] = 0

    labeled = label(labeled > 0)  # update num of Part3
    num = int(np.max(labeled))

    # process for Part2
    kernel_p = cube(5, decomposition="sequence")
    post_pro = binary_opening(post_pro, footprint=kernel_p)
    labeled_post_pro = label(post_pro)
    num_post = int(np.max(labeled_post_pro))

    labeled_post_pro += num
    labeled_post_pro[labeled_post_pro == num] = 0  # update num of Part2

    num += num_post  # update total num of vesicles(except pre_pro)
    labeled_pre_pro += num
    labeled_pre_pro[labeled_pre_pro == num] = 0  # update num of label for part 1
    labeled = labeled + labeled_post_pro + labeled_pre_pro

    # for main vesicles
    filtered = labeled >= 1
    logging.info("\ncomplete filtering\n")
    boundaries = filtered & ~binary_erosion(
        filtered, footprint=cube(3, decomposition="sequence")
    )

    bd_labeled = label(boundaries)
    vesicle_list = _points_by_label_bboxes(bd_labeled)

    return vesicle_list, shape


def density_fit(data_iso, center, radius, pad_value=None):
    """input center(z,y,x), output center(z,y.x), both in array"""
    # padwidth = int(max(-min(center-radius), -min(np.array(shape)-1-center-radius),0))+5
    if pad_value is None:
        pad_value = np.max(data_iso)
    cube_, center = _extract_padded_cube(data_iso, center, radius, pad_value)
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
    vesicle_points = np.argwhere(labeled > 0)
    [center_cube, evecs, radii] = ef.ellipsoid_fit(vesicle_points)
    if np.min(center_cube) < 0:  # if the shape of fitted ellipsoid is too strange
        return [None, None, None, 0]

    tm = template(radii, center_cube, evecs, cube_.shape)
    ccf = CCF(cube_normalize, tm)
    [center_fit, evecs_fit, radii_fit] = [
        center + center_cube - cube_.shape[0] // 2,
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
    Generate a 2D Gaussian weight matrix.
    The weights are highest at the center and decrease towards the edges, following a Gaussian distribution.
    @param sigma: controls the decay rate of the Gaussian distribution
    """
    center = size // 2
    x, y = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    x_centered = x - center
    y_centered = y - center
    squared_dist = x_centered**2 + y_centered**2
    gauss = np.exp(-squared_dist / (2 * sigma**2))
    gauss /= gauss.max()
    return gauss


def density_fit_2d(data_iso, center, radius, pad_value=None):
    """input center(z,y,x), output center(z,y.x), both in array"""

    # padwidth = int(max(-min(center-radius), -min(np.array(shape)-1-center-radius),0))+5
    if pad_value is None:
        pad_value = np.max(data_iso)
    cube_, center = _extract_padded_cube(data_iso, center, radius, pad_value)
    # cube_2=data_pad[center[0]-int(radius)-15: center[0]+int(radius)+15+1,center[1]-int(radius)-15: center[1]+int(radius)+15+1,center[2]-int(radius)-15: center[2]+int(radius)+15+1]

    img = cube_[cube_.shape[0] // 2, :, :]
    img = ndimage.gaussian_filter(img, sigma=1)

    img_reverse = -img.astype(np.float32)
    img_normalize = (img_reverse - np.min(img_reverse)) / (
        np.max(img_reverse) - np.min(img_reverse)
    )

    sigma = int(radius) + 5
    gaussian_weights = generate_2d_gaussian_weights(img.shape[0], sigma)
    img_normalize = img_normalize * gaussian_weights
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
        center + center_cube - cube_.shape[0] // 2,
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


def measure_one(idx, data=None, vesicle_list=None, min_radius=None, data_max=None):
    if data is None:
        data = _MEASURE_DATA
    if vesicle_list is None:
        vesicle_list = _MEASURE_VESICLE_LIST
    if min_radius is None:
        min_radius = _MEASURE_MIN_RADIUS
    if data_max is None:
        data_max = _MEASURE_DATA_MAX

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

    [center, evecs, radii, ccf] = density_fit(data, center0, np.max(radii), data_max)
    if ccf < 0.3:  # delete wrong segments
        return

    if if_normal(radii):
        info = {
            "name": "vesicle_" + str(idx + 1),
            "center": center.tolist(),
            "radii": radii.tolist(),
            "evecs": evecs.tolist(),
            "CCF": str(ccf),
        }
        return info


def vesicle_measure(data, vesicle_list, shape, min_radius, outfile=None):
    """ """
    results = []
    global in_count
    global sup_in_count
    in_count = 0
    sup_in_count = 0

    logging.info("\nStart vesicle measurement\n")

    idxs = range(len(vesicle_list))
    if len(vesicle_list) == 0:
        vesicle_info = {"vesicles": results}
        if outfile is not None:
            with open(outfile, "w") as out:
                json.dump(vesicle_info, out)
        return vesicle_info

    data_max = np.max(data)
    poolnum = min(multiprocessing.cpu_count(), 4)
    with multiprocessing.Pool(
        poolnum,
        initializer=_init_measure_worker,
        initargs=(data, vesicle_list, min_radius, data_max),
    ) as pool:
        results = list(
            tqdm(
                pool.imap(measure_one, idxs),
                total=len(vesicle_list),
                file=sys.stdout,
            )
        )
    results = list(filter(None, results))

    # return vesicle information dict and save as json
    vesicle_info = {"vesicles": results}

    if outfile is not None:
        with open(outfile, "w") as out:
            json.dump(vesicle_info, out)
    
    return vesicle_info


def _render_dense_points(vesicle_tomo, points, label_id):
    if points.size == 0:
        return
    zmin, ymin, xmin = np.min(points, axis=0)
    zmax, ymax, xmax = np.max(points, axis=0)
    offset = np.array([zmin, ymin, xmin])
    local_points = points - offset
    local_shape = (zmax - zmin + 1, ymax - ymin + 1, xmax - xmin + 1)
    local_mask = np.zeros(local_shape, dtype=bool)
    local_mask[local_points[:, 0], local_points[:, 1], local_points[:, 2]] = True
    local_mask = closing(local_mask, cube(3))
    z, y, x = np.nonzero(local_mask)
    vesicle_tomo[z + zmin, y + ymin, x + xmin] = label_id


def vesicle_rendering(vesicle_file, tomo_dims):
    """ """
    # vesicle file can be json path, {"vesicles": [...]}, or a vesicle info list.
    if type(vesicle_file) is str:
        with open(vesicle_file) as f:
            ves = json.load(f)
        vesicle_info = ves["vesicles"]
    elif isinstance(vesicle_file, dict):
        vesicle_info = vesicle_file["vesicles"]
    else:
        vesicle_info = vesicle_file
    tomo_shape = tuple(np.asarray(tomo_dims, dtype=np.int64))
    vesicle_tomo = np.zeros(tomo_shape, dtype=np.int16)
    # vesicle_tomo = np.zeros(np.array(tomo_dims),dtype=np.uint8)
    logging.info("\nrendering vesicle\n")
    # for i,vesicle in enumerate(vesicle_info):
    for i in tqdm(range(len(vesicle_info)), file=sys.stdout):
        ellip_i = mk.ellipsoid_point_dense(
            vesicle_info[i]["radii"],
            vesicle_info[i]["center"],
            vesicle_info[i]["evecs"],
            tomo_shape,
        )
        Id = int(vesicle_info[i]["name"][8:])
        _render_dense_points(vesicle_tomo, ellip_i, Id)

    # vesicle_tomo = closing(vesicle_tomo,cube(3))
    logging.info("{} vesicles in total".format(len(vesicle_info)))
    return vesicle_tomo


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

    # save raw vesicle mask
    with mrcfile.open(args.mask_file) as m:
        bimask = m.data
    shape = bimask.shape
    print("begin morph process")
    vesicle_list, shape = morph_process(
        args.mask_file, args.area_file, pixelsize=args.pixelsize, radius=args.min_radius
    )
    print("done morph process")

    with mrcfile.mmap(args.tomo_file, mode="r", permissive=True) as m:
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
