#!/usr/bin/env python

import os
import json
import fire
import napari
import mrcfile
import threading
import numpy as np

from qtpy import QtCore, QtWidgets
from scipy.spatial import KDTree
from skimage.morphology import closing, cube
from napari import Viewer
from napari.settings import get_settings
from napari.resources import ICONS
from napari.utils.notifications import show_info
from napari._qt.widgets.qt_viewer_buttons import QtViewerPushButton

from enum import Enum
from three_orthos_viewer import CrossWidget, MultipleViewerWidget
from segVesicle.utils import make_ellipsoid as mk
from morph import density_fit, density_fit_2d, fit_6pts, dis
from global_vars import TOMO_SEGMENTATION_PROGRESS, TomoPath
import center_cross

def print_in_widget(message):
    if dock_widget:
        dock_widget.message_signal.emit(message)

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
        print_in_widget("Delete label.")

def create_delete_button(viewer):
    '''Creates a delete button with the specified icon and inserts it into the viewer's layout'''
    del_button = QtViewerPushButton('delete label')
    del_icon_path = ICONS.get('delete')
    del_icon = change_icon_color(del_icon_path, 'yellow')
    del_button.setIcon(del_icon)
    
    # 将按钮插入到布局中
    layer_buttons = viewer.window.qt_viewer.layerButtons
    layer_buttons.layout().insertWidget(6, del_button)
    
    return del_button

def register_save_shortcut_delete(viewer, root_dir, new_json_file_path):
    '''press 'd' to save the point to delete and save the new label layer
    '''
    @viewer.bind_key('d', overwrite=True)
    def save_label_image(viewer):
        threading.Thread(target=save_and_update_delete, args=(viewer, root_dir, new_json_file_path)).start()
    del_button = create_button(viewer, 'delete label', 'delete', 'yellow', 6)
    del_button.clicked.connect(lambda: save_label_image(viewer))


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
        print_in_widget("Add 3d label.")

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
        print_in_widget("Add 2d label.")

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
        print_in_widget("Add 6pts label.")

def register_save_shortcut_add(viewer, root_dir, new_json_file_path):
    @viewer.bind_key('g', overwrite=True)
    def save_point_image(viewer):
        threading.Thread(target=save_and_update_add, args=(viewer, root_dir, new_json_file_path)).start()
    # 创建添加按钮并将其点击事件与 save_point_image 函数绑定
    add_button = create_button(viewer, 'add 3D label', 'add', 'yellow', 3)
    add_button.clicked.connect(lambda: save_point_image(viewer))
    
def register_save_shortcut_add_2d(viewer, root_dir, new_json_file_path):
    @viewer.bind_key('f', overwrite=True)
    def save_point_image(viewer):
        threading.Thread(target=save_and_update_add_2d, args=(viewer, root_dir, new_json_file_path)).start()
    # 创建添加 2D 按钮并将其点击事件与 save_point_image 函数绑定
    add_2d_button = create_button(viewer, 'add 2D label', 'add', 'white', 4)
    add_2d_button.clicked.connect(lambda: save_point_image(viewer))


def register_save_shortcut_add_6pts(viewer, root_dir, new_json_file_path):
    @viewer.bind_key('p', overwrite=True)
    def save_point_image(viewer):
        threading.Thread(target=save_and_update_add_6pts, args=(viewer, root_dir, new_json_file_path)).start()
    # 创建添加 6pts 按钮并将其点击事件与 save_and_update_add_6pts 函数绑定
    add_6pts_button = create_button(viewer, 'add 6pts label', 'polygon_lasso', 'yellow', 5)
    add_6pts_button.clicked.connect(lambda: threading.Thread(target=save_and_update_add_6pts, args=(viewer, root_dir, new_json_file_path)).start())


def create_button(viewer, label, icon_key, icon_color, position):
    '''Creates a button with the specified label, icon, and color, and inserts it into the viewer's layout'''
    button = QtViewerPushButton(label)
    icon_path = ICONS.get(icon_key)
    icon = change_icon_color(icon_path, icon_color)
    button.setIcon(icon)
    
    # 将按钮插入到布局中
    layer_buttons = viewer.window.qt_viewer.layerButtons
    layer_buttons.layout().insertWidget(position, button)
    
    return button

def change_icon_color(icon_path, color):
    from qtpy.QtGui import QPixmap, QPainter, QColor, QIcon
    pixmap = QPixmap(icon_path)
    painter = QPainter(pixmap)
    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.fillRect(pixmap.rect(), QColor(color))
    painter.end()
    return QIcon(pixmap)

def add_button_and_register_add_and_delete(viewer: Viewer, root_dir, new_json_file_path):
    register_save_shortcut_delete(viewer, root_dir, new_json_file_path)
    register_save_shortcut_add(viewer, root_dir, new_json_file_path)
    register_save_shortcut_add_2d(viewer, root_dir, new_json_file_path)
    register_save_shortcut_add_6pts(viewer, root_dir, new_json_file_path)
    
    layer_buttons = viewer.window.qt_viewer.layerButtons

    # 删除位置在1，2，3的按钮
    for i in [2, 1, 0]:
        item = layer_buttons.layout().takeAt(i)
        if item is not None:
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

def main(tomo_dir):
    pid = os.getpid()
    root_dir = os.path.abspath('temp') + '/'
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    tomo_path=TomoPath(tomo_dir, root_dir, pid)

    os.system('cp {} {}'.format(tomo_path.json_file_path, tomo_path.new_json_file_path))
    if not os.path.exists(tomo_path.ori_label_path):
        os.system('cp {} {}'.format(tomo_path.label_path, tomo_path.ori_label_path))
        os.system('cp {} {}'.format(tomo_path.json_file_path, tomo_path.ori_json_file_path))
    
    # calculate contrast limits
    lambda_scale = 0.35
    tomo = get_tomo(tomo_path.isonet_tomo_path)
    mi, ma = (tomo.max() - tomo.min()) * lambda_scale + tomo.min(), tomo.max() - (tomo.max() - tomo.min()) * lambda_scale

    # change increment dims shortcuts
    settings = get_settings()
    settings.shortcuts.shortcuts['napari:increment_dims_left'] = ['PageUp']
    settings.shortcuts.shortcuts['napari:increment_dims_right'] = ['PageDown']
    # set default interface
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    # viewer = Viewer()
    viewer = Viewer()
    main_viewer = viewer.window.qt_viewer.parentWidget()
    global dock_widget
    dock_widget = MultipleViewerWidget(viewer)
    cross = CrossWidget(viewer)
    main_viewer.layout().addWidget(dock_widget)
    viewer.window.add_dock_widget(cross, name="Cross", area="left")
    
    viewer.add_labels(get_tomo(tomo_path.label_path).astype(np.int16), name='label')  # add label layer
    viewer.add_image(get_tomo(tomo_path.isonet_tomo_path), name='corrected_tomo')  # add isonet treated tomogram layer
    viewer.add_points(name='edit vesicles', ndim=3, size=4)  # add an empty Points layer
    
    # ls： The window will not automatically adjust for now; manually zoom to set an appropriate value
    dock_widget.viewer_model1.camera.zoom = 1.95
    dock_widget.viewer_model2.camera.zoom = 1.5
    
    viewer.layers['corrected_tomo'].opacity = 0.5
    viewer.layers['corrected_tomo'].contrast_limits = [mi, ma]
    viewer.layers['edit vesicles'].mode = 'ADD'

    print_in_widget("Welcome to the Vesicle Segmentation Software, version 0.1.")
    print_in_widget("For instructions and keyboard shortcuts, please refer to the help documentation available in the '?' section at the top right corner.")
    
    add_button_and_register_add_and_delete(viewer, root_dir, tomo_path.new_json_file_path)
    
    napari.run()
    
    os.system('mv {} {}'.format(tomo_path.new_json_file_path, tomo_path.json_file_path))
    os.system('mv {} {}'.format(tomo_path.new_label_file_path, tomo_path.label_path))
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
