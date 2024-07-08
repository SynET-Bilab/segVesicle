import os
import json
import mrcfile
import threading
import numpy as np

from qtpy.QtCore import QObject
from qtpy.QtGui import QTextCursor
from skimage.morphology import closing, cube
from scipy.spatial import KDTree
from napari.utils.notifications import show_info
from napari.resources import ICONS
# from napari._qt.widgets.qt_viewer_buttons import QtViewerPushButton

from segVesicle.utils import make_ellipsoid as mk
from morph import density_fit, density_fit_2d, fit_6pts, dis
from tomo_viewer import TomoViewer

# qRegisterMetaType(QTextCursor)

LABEL_START = 10000  # large enough to avoid overlap with original labe
LABEL_LAYER_IDX = 'label'
POINT_LAYER_IDX = 'edit vesicles'
NUM_POINT = 0
global added_vesicle_num
added_vesicle_num = 0
label_history = None
tomo_path = None

def get_tomo(path):
    with mrcfile.open(path) as mrc:
        data = mrc.data
    data = np.flip(data, axis=1)
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

def save_point_layer(tomo_viewer, layer_idx, mode):
    global added_vesicle_num
    viewer = tomo_viewer.viewer
    
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

def save_label_layer(tomo_viewer, layer_idx):
    viewer = tomo_viewer.viewer
    root_dir = tomo_viewer.tomo_path_and_stage.root_dir
    save_path = root_dir + 'label_{}.mrc'.format(os.getpid())
    if len(viewer.layers) > 0:
        image_layer = viewer.layers[layer_idx]
        data = np.asarray(image_layer.data).astype(np.float32)
        data = np.flip(data, axis=1)
        with mrcfile.new(save_path, overwrite=True) as mrc:
            mrc.set_data(data)
    show_info('Saved at {}'.format(os.path.abspath(save_path)))

def get_info_from_json(json_file):
    with open(json_file, "r") as f:
        vesicles = json.load(f)['vesicles']
    centers = []
    for vesicle in vesicles:
        centers.append(vesicle['center'])
    centers = np.asarray(centers)
    tree = KDTree(centers, leafsize=2)
    return vesicles, tree

def update_json_file(tomo_viewer, point, mode, vesicle_to_add):
    json_file = tomo_viewer.tomo_path_and_stage.new_json_file_path
    vesicles, tree = get_info_from_json(json_file)
    if mode == 'Deleted':
        delete_idx = tree.query(point[0].reshape(1, -1), k=1)[1][0]  # delete mode has only 1 point at idx 0
        vesicles.pop(delete_idx)
    elif mode == 'Added':
        vesicles.append(vesicle_to_add)
    vesicle_info = {'vesicles': vesicles}
    with open(json_file, "w", encoding='utf-8') as out:
        json.dump(vesicle_info, out)

def add_vesicle_show(tomo_viewer, point, add_mode):
    viewer = tomo_viewer.viewer
    ori_tomo = viewer.layers['corrected_tomo'].data
    label_idx = LABEL_START + added_vesicle_num
    data_to_add, new_added_vesicle = add_vesicle(ori_tomo, point, label_idx, add_mode)
    return data_to_add.astype(np.int16), new_added_vesicle

def delete_picked_vesicle(tomo_viewer, deleted_point):
    viewer = tomo_viewer.viewer
    z, y, x = int(deleted_point[0][0]), int(deleted_point[0][1]), int(deleted_point[0][2])
    label_num = viewer.layers[LABEL_LAYER_IDX].data[z, y, x]
    label_layer_data = np.asarray(viewer.layers[LABEL_LAYER_IDX].data)
    label_layer_data[label_layer_data == label_num] = 0
    viewer.layers[LABEL_LAYER_IDX].data = label_layer_data
    viewer.layers[LABEL_LAYER_IDX].refresh()

def add_picked_vesicle(tomo_viewer, data_to_add):
    viewer = tomo_viewer.viewer
    if np.sum(np.sign(viewer.layers[LABEL_LAYER_IDX].data) * np.sign(data_to_add)) > 0:
        show_info('Please reselect two points')
    else:
        viewer.layers[LABEL_LAYER_IDX].data = viewer.layers[LABEL_LAYER_IDX].data + data_to_add  # update label layer
        viewer.layers[LABEL_LAYER_IDX].refresh()

def save_and_update_delete(tomo_viewer):
    viewer = tomo_viewer.viewer
    if len(viewer.layers[POINT_LAYER_IDX].data) < 1:
        show_info('Please pick a point to delete')
        tomo_viewer.print('Please pick a point to delete')
    else:
        point = save_point_layer(tomo_viewer, POINT_LAYER_IDX, mode='Deleted')
        delete_picked_vesicle(tomo_viewer, point)
        viewer.layers[POINT_LAYER_IDX].data = None
        save_label_layer(tomo_viewer, LABEL_LAYER_IDX)
        update_json_file(tomo_viewer, point, mode='Deleted', vesicle_to_add=None)
        tomo_viewer.print('Successfully deleted Vesicle')

def create_delete_button(tomo_viewer):
    from napari._qt.widgets.qt_viewer_buttons import QtViewerPushButton
    viewer = tomo_viewer.viewer
    del_button = QtViewerPushButton('delete label')
    del_icon_path = ICONS.get('delete')
    del_icon = change_icon_color(del_icon_path, 'yellow')
    del_button.setIcon(del_icon)
    
    layer_buttons = viewer.window.qt_viewer.layerButtons
    layer_buttons.layout().insertWidget(6, del_button)
    
    return del_button

def register_save_shortcut_delete(tomo_viewer):
    viewer = tomo_viewer.viewer
    @viewer.bind_key('d', overwrite=True)
    def save_label_image(viewer):
        threading.Thread(target=save_and_update_delete, args=(tomo_viewer,)).start()
    del_button = create_button(viewer, 'Delete label (Shortcut: d)', 'delete', 'yellow', 6)
    del_button.clicked.connect(lambda: save_label_image(viewer))

def save_and_update_add(tomo_viewer):
    viewer = tomo_viewer.viewer
    if len(viewer.layers[POINT_LAYER_IDX].data) < 2:
        show_info('Please add two points to define a vesicle')
        tomo_viewer.print('Please add two points to define a vesicle')
    else:
        point = save_point_layer(tomo_viewer, POINT_LAYER_IDX, mode='Added')
        data_to_add, new_added_vesicle = add_vesicle_show(tomo_viewer, point, add_mode='3d')
        add_picked_vesicle(tomo_viewer, data_to_add)
        viewer.layers[POINT_LAYER_IDX].data = None
        save_label_layer(tomo_viewer, LABEL_LAYER_IDX)
        update_json_file(tomo_viewer, point, mode='Added', vesicle_to_add=new_added_vesicle[0])
        tomo_viewer.print('Successfully added 3d Vesicle')

def save_and_update_add_2d(tomo_viewer):
    viewer = tomo_viewer.viewer
    if len(viewer.layers[POINT_LAYER_IDX].data) < 2:
        show_info('Please add two points to define a vesicle')
        tomo_viewer.print('Please add two points to define a vesicle')
    else:
        point = save_point_layer(tomo_viewer, POINT_LAYER_IDX, mode='Added')
        data_to_add, new_added_vesicle = add_vesicle_show(tomo_viewer, point, add_mode='2d')
        add_picked_vesicle(tomo_viewer, data_to_add)
        viewer.layers[POINT_LAYER_IDX].data = None
        save_label_layer(tomo_viewer, LABEL_LAYER_IDX)
        update_json_file(tomo_viewer, point, mode='Added', vesicle_to_add=new_added_vesicle[0])
        tomo_viewer.print('Successfully added 2d Vesicle')

def save_and_update_add_6pts(tomo_viewer):
    viewer = tomo_viewer.viewer
    if len(viewer.layers[POINT_LAYER_IDX].data) < 6:
        show_info('Please add 6 points to fit a vesicle')
        tomo_viewer.print('Please add 6 points to fit a vesicle')
    else:
        point = save_point_layer(tomo_viewer, POINT_LAYER_IDX, mode='Added_6pts')
        data_to_add, new_added_vesicle = add_vesicle_show(tomo_viewer, point, add_mode='6pts')
        add_picked_vesicle(tomo_viewer, data_to_add)
        viewer.layers[POINT_LAYER_IDX].data = None
        save_label_layer(tomo_viewer, LABEL_LAYER_IDX)
        update_json_file(tomo_viewer, point, mode='Added', vesicle_to_add=new_added_vesicle[0])
        tomo_viewer.print('Successfully added 2d Vesicle')

def register_save_shortcut_add(tomo_viewer):
    viewer = tomo_viewer.viewer
    @viewer.bind_key('g', overwrite=True)
    def save_point_image(viewer):
        threading.Thread(target=save_and_update_add, args=(tomo_viewer,)).start()
    add_button = create_button(viewer, 'Add 3D label (Shortcut: g)', 'add', 'yellow', 0)
    add_button.clicked.connect(lambda: save_point_image(viewer))

def register_save_shortcut_add_2d(tomo_viewer):
    viewer = tomo_viewer.viewer
    @viewer.bind_key('f', overwrite=True)
    def save_point_image(viewer):
        threading.Thread(target=save_and_update_add_2d, args=(tomo_viewer,)).start()
    add_2d_button = create_button(viewer, 'Add 2D label (Shortcut: f)', 'add', 'white', 1)
    add_2d_button.clicked.connect(lambda: save_point_image(viewer))

def register_save_shortcut_add_6pts(tomo_viewer):
    viewer = tomo_viewer.viewer
    @viewer.bind_key('p', overwrite=True)
    def save_point_image(viewer):
        threading.Thread(target=save_and_update_add_6pts, args=(tomo_viewer,)).start()
    add_6pts_button = create_button(viewer, 'Add 6pts label (Shortcut: p)', 'polygon_lasso', 'yellow', 2)
    add_6pts_button.clicked.connect(lambda: threading.Thread(target=save_and_update_add_6pts, args=(tomo_viewer,)).start())

def create_button(viewer, label, icon_key, icon_color, position):
    from napari._qt.widgets.qt_viewer_buttons import QtViewerPushButton
    button = QtViewerPushButton(label)
    icon_path = ICONS.get(icon_key)
    icon = change_icon_color(icon_path, icon_color)
    button.setIcon(icon)
    
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


def add_button_and_register_add_and_delete(tomo_viewer: TomoViewer):
    layer_buttons = tomo_viewer.viewer.window.qt_viewer.layerButtons

    for i in [4, 2, 1, 0]:
        item = layer_buttons.layout().takeAt(i)
        if item is not None:
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
    
    register_shortcuts_and_buttons(tomo_viewer, [
        register_save_shortcut_add,
        register_save_shortcut_add_2d,
        register_save_shortcut_add_6pts,
        register_save_shortcut_delete
    ])

def register_shortcuts_and_buttons(tomo_viewer: TomoViewer, register_functions: list):
    for func in register_functions:
        func(tomo_viewer)
