#!/usr/bin/env python

import os
import json
import fire
import napari
import mrcfile
import threading
import numpy as np

from scipy.spatial import KDTree
from skimage.morphology import closing, cube
from napari.utils.notifications import show_info

from segVesicle.utils import make_ellipsoid as mk
from morph import density_fit, density_fit_2d, fit_6pts, dis



def get_tomo(path):
    with mrcfile.open(path) as mrc:
        data = mrc.data
    return data


def vesicle_rendering(vesicle_info, tomo_dims, idx):
    
    vesicle_tomo = np.zeros(np.array(tomo_dims) + np.array([30,30,30]), dtype=np.int16)

    for i in range(len(vesicle_info)):
        ellip_i = mk.ellipsoid_point(vesicle_info[i]['radii'], vesicle_info[i]['center'], vesicle_info[i]['evecs'])
        vesicle_tomo[ellip_i[:,0], ellip_i[:,1], ellip_i[:,2]] = idx
        xmin, xmax = np.min(ellip_i[:,2]), np.max(ellip_i[:,2])
        ymin, ymax = np.min(ellip_i[:,1]), np.max(ellip_i[:,1])
        zmin, zmax = np.min(ellip_i[:,0]), np.max(ellip_i[:,0])
        cube_i = vesicle_tomo[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
        cube_i = closing(cube_i, cube(3))
        vesicle_tomo[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1] = cube_i
    return vesicle_tomo[0:tomo_dims[0], 0:tomo_dims[1], 0:tomo_dims[2]]


def add_vesicle(data_iso, points, label_idx, add_mode = '3d'):
    
    shape = data_iso.shape
    center_manual = points[0]
    radius_point = points[1]

    radius = dis(center_manual, radius_point)
    if add_mode == '3d':
        [center, evecs, radii, ccf] = density_fit(data_iso, center_manual, radius)
    elif add_mode == '2d':
        [center, evecs, radii, ccf] = density_fit_2d(data_iso, center_manual, radius)
    elif add_mode == '6pts':
        [center, evecs, radii, ccf] = fit_6pts(data_iso, points)
    info = {'name':'vesicle_'+str(label_idx), 'center':center.tolist(), 'radii':radii.tolist(), 'evecs':evecs.tolist(), 'CCF':str(ccf)}
    
    result_to_show = [info]
    ves_tomo = vesicle_rendering(result_to_show, shape, label_idx).astype(np.int16)
    
    return ves_tomo, result_to_show


def save_point_layer(viewer, layer_idx, mode):
    
    global added_vesicle_num
    
    if len(viewer.layers) > 0:
        image_layer = viewer.layers[layer_idx]
        if mode == 'Deleted':
            image_data = image_layer.data[-1:]
        elif mode == 'Added':
            image_data = image_layer.data[-2:]
            added_vesicle_num += 1  # to label new added vesicles
        elif mode == 'Added_6pts':
            image_data = image_layer.data[-6:]
            added_vesicle_num += 1
    else:
        show_info('no points added')
    
    return image_data


def save_label_layer(viewer, root_dir, layer_idx):
    '''save label layer as new labels
    '''
    save_path = root_dir + 'label_{}.mrc'.format(os.getpid())
    if len(viewer.layers) > 0:
        image_layer = viewer.layers[layer_idx]
        with mrcfile.new(save_path, overwrite=True) as mrc:
            mrc.set_data(np.asarray(image_layer.data).astype(np.float32))
    show_info('Saved at {}'.format(os.path.abspath(save_path)))
    

def get_info_from_json(json_file):
    '''
    '''
    with open(json_file, "r") as f:
        vesicles = json.load(f)['vesicles']
    centers = []
    for vesicle in vesicles:
        centers.append(vesicle['center'])
    centers = np.asarray(centers)
    tree = KDTree(centers, leafsize=2)
    return vesicles, tree


def update_json_file(viewer, point, json_file, mode, vesicle_to_add):
    '''
    '''
    vesicles, tree = get_info_from_json(json_file)
    if mode == 'Deleted':
        delete_idx = tree.query(point[0].reshape(1, -1), k=1)[1][0]  # delete mode has only 1 point at idx 0
        vesicles.pop(delete_idx)
    elif mode == 'Added':
        vesicles.append(vesicle_to_add)
    vesicle_info = {'vesicles': vesicles}
    with open(json_file, "w", encoding='utf-8') as out:
        json.dump(vesicle_info, out)


def add_vesicle_show(viewer, point, add_mode):
    '''calculate the added vesicle
    '''
    ori_tomo = viewer.layers[-2].data
    
    label_idx = LABEL_START + added_vesicle_num
    data_to_add, new_added_vesicle = add_vesicle(ori_tomo, point, label_idx, add_mode)
    return data_to_add.astype(np.int16), new_added_vesicle


def delete_picked_vesicle(viewer, deleted_point):
    z, y, x = int(deleted_point[0][0]), int(deleted_point[0][1]), int(deleted_point[0][2])
    label_num = viewer.layers[LABEL_LAYER_IDX].data[z, y, x]
    label_layer_data = np.asarray(viewer.layers[LABEL_LAYER_IDX].data)
    label_layer_data[label_layer_data == label_num] = 0
    viewer.layers[LABEL_LAYER_IDX].data = label_layer_data
    viewer.layers[LABEL_LAYER_IDX].refresh()


def add_picked_vesicle(viewer, data_to_add):
    if np.sum(np.sign(viewer.layers[LABEL_LAYER_IDX].data) * np.sign(data_to_add)) > 0:
        show_info('Please reselect two points')
    else:
        viewer.layers[LABEL_LAYER_IDX].data = viewer.layers[LABEL_LAYER_IDX].data + data_to_add  # update label layer
        viewer.layers[LABEL_LAYER_IDX].refresh()


def save_and_update_delete(viewer, root_dir, new_json_file_path):
    if len(viewer.layers[POINT_LAYER_IDX].data) < 1:
        show_info('Please pick a point to delete')
    else:
        point = save_point_layer(viewer, POINT_LAYER_IDX, mode='Deleted')
        delete_picked_vesicle(viewer, point)
        viewer.layers[POINT_LAYER_IDX].data = None
        save_label_layer(viewer, root_dir, LABEL_LAYER_IDX)
        update_json_file(viewer, point, new_json_file_path, mode='Deleted', vesicle_to_add=None)


def register_save_shortcut_delete(viewer, root_dir, new_json_file_path):
    '''press 'd' to save the point to delete and save the new label layer
    '''
    @viewer.bind_key('d', overwrite=True)
    def save_label_image(viewer):
        threading.Thread(target=save_and_update_delete, args=(viewer, root_dir, new_json_file_path)).start()


def save_and_update_add(viewer, root_dir, new_json_file_path):
    if len(viewer.layers[POINT_LAYER_IDX].data) < 2:
        show_info('Please add two points to define a vesicle')
    else:
        point = save_point_layer(viewer, POINT_LAYER_IDX, mode='Added')
        data_to_add, new_added_vesicle = add_vesicle_show(viewer, point, add_mode='3d')
        add_picked_vesicle(viewer, data_to_add)
        viewer.layers[POINT_LAYER_IDX].data = None
        save_label_layer(viewer, root_dir, LABEL_LAYER_IDX)
        update_json_file(viewer, point, new_json_file_path, mode='Added', vesicle_to_add=new_added_vesicle[0])

def save_and_update_add_2d(viewer, root_dir, new_json_file_path):
    if len(viewer.layers[POINT_LAYER_IDX].data) < 2:
        show_info('Please add two points to define a vesicle')
    else:
        point = save_point_layer(viewer, POINT_LAYER_IDX, mode='Added')
        data_to_add, new_added_vesicle = add_vesicle_show(viewer, point, add_mode='2d')
        add_picked_vesicle(viewer, data_to_add)
        viewer.layers[POINT_LAYER_IDX].data = None
        save_label_layer(viewer, root_dir, LABEL_LAYER_IDX)
        update_json_file(viewer, point, new_json_file_path, mode='Added', vesicle_to_add=new_added_vesicle[0])

def save_and_update_add_6pts(viewer, root_dir, new_json_file_path):
    if len(viewer.layers[POINT_LAYER_IDX].data) < 6:
        show_info('Please add 6 points to fit a vesicle')
    else:
        point = save_point_layer(viewer, POINT_LAYER_IDX, mode='Added_6pts')
        data_to_add, new_added_vesicle = add_vesicle_show(viewer, point, add_mode='6pts')
        add_picked_vesicle(viewer, data_to_add)
        viewer.layers[POINT_LAYER_IDX].data = None
        save_label_layer(viewer, root_dir, LABEL_LAYER_IDX)
        update_json_file(viewer, point, new_json_file_path, mode='Added', vesicle_to_add=new_added_vesicle[0])

def register_save_shortcut_add(viewer, root_dir, new_json_file_path):
    @viewer.bind_key('g', overwrite=True)
    def save_point_image(viewer):
        threading.Thread(target=save_and_update_add, args=(viewer, root_dir, new_json_file_path)).start()

def register_save_shortcut_add_2d(viewer, root_dir, new_json_file_path):
    @viewer.bind_key('f', overwrite=True)
    def save_point_image(viewer):
        threading.Thread(target=save_and_update_add_2d, args=(viewer, root_dir, new_json_file_path)).start()

def register_save_shortcut_add_6pts(viewer, root_dir, new_json_file_path):
    @viewer.bind_key('p', overwrite=True)
    def save_point_image(viewer):
        threading.Thread(target=save_and_update_add_6pts, args=(viewer, root_dir, new_json_file_path)).start()

def main(tomo_dir):
    pid = os.getpid()
    root_dir = os.path.abspath('temp') + '/'
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    ori_tomo_path = os.path.abspath(tomo_dir + '_wbp.mrc')
    deconv_tomo_path = os.path.abspath('tomoset/' + tomo_dir + '_dec.mrc')
    isonet_tomo_path = os.path.abspath(tomo_dir + '_wbp_corrected.mrc')
    segment_path = os.path.abspath(tomo_dir + '_segment.mrc')
    label_path = os.path.abspath(tomo_dir + '_label_vesicle.mrc')
    json_file_path = os.path.abspath(tomo_dir + '_vesicle.json')
    new_json_file_path = root_dir + 'vesicle_new_{}.json'.format(pid)
    new_label_file_path = root_dir + 'label_{}.mrc'.format(pid)
    final_json_file_path = os.path.abspath(tomo_dir + '_vesicle_final.json')
    final_label_path = os.path.abspath(tomo_dir + '_label_vesicle_final.mrc')
    
    cmd_cp = 'cp {} {}'.format(json_file_path, new_json_file_path)
    os.system(cmd_cp)
    
    # calculate contrast limits
    lambda_scale = 0.35
    tomo = get_tomo(isonet_tomo_path)
    mi, ma = (tomo.max() - tomo.min()) * lambda_scale + tomo.min(), tomo.max() - (tomo.max() - tomo.min()) * lambda_scale

    # set default interface
    viewer = napari.Viewer()
    
    viewer.add_labels(get_tomo(label_path).astype(np.int16), name='label')  # add label layer
    viewer.add_image(get_tomo(isonet_tomo_path), name='corrected_tomo')  # add isonet treated tomogram layer
    viewer.add_points(name='edit vesicles', ndim=3, size=4)  # add an empty Points layer
    
    viewer.layers['corrected_tomo'].opacity = 0.5
    viewer.layers['corrected_tomo'].contrast_limits = [mi, ma]
    viewer.layers['edit vesicles'].mode = 'ADD'

    register_save_shortcut_delete(viewer, root_dir, new_json_file_path)
    register_save_shortcut_add(viewer, root_dir, new_json_file_path)
    register_save_shortcut_add_2d(viewer, root_dir, new_json_file_path)
    register_save_shortcut_add_6pts(viewer, root_dir, new_json_file_path)
    napari.run()
    
    os.system('mv {} {}'.format(new_json_file_path, final_json_file_path))
    os.system('cp {} {}'.format(final_json_file_path, json_file_path))
    os.system('mv {} {}'.format(new_label_file_path, final_label_path))
    os.system('cp {} {}'.format(final_label_path, label_path))
    os.system('rm -r {}'.format(root_dir))


if __name__ == '__main__':
    
    # set default params
    LABEL_START = 10000  # large enough to avoid overlap with original label
    LABEL_LAYER_IDX = 0
    POINT_LAYER_IDX = 2
    NUM_POINT = 0
    global added_vesicle_num
    added_vesicle_num = 0

    fire.Fire(main)
