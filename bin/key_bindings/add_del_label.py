import os
import json
import mrcfile
import threading
import numpy as np
import queue
import tempfile
import traceback
import re

from qtpy.QtCore import QObject
from qtpy.QtGui import QTextCursor
from skimage.morphology import closing, cube
from napari.utils.notifications import show_info
from napari.resources import ICONS
# from napari._qt.widgets.qt_viewer_buttons import QtViewerPushButton

from util.io import get_tomo
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

# 全局任务队列和锁
task_queue = queue.Queue()
lock = threading.Lock()
VESICLE_NAME_PATTERN = re.compile(r"^vesicle_(\d+)$")

# 处理队列中的任务
def process_queue():
    while True:
        func, args = task_queue.get()
        try:
            func(*args)
        except Exception:
            print(f"[add_del_label] Queued task {func.__name__} failed")
            traceback.print_exc()
        finally:
            task_queue.task_done()

# 启动队列处理线程
queue_thread = threading.Thread(target=process_queue, daemon=True)
queue_thread.start()

# 以队列方式执行的操作
def save_and_update_add_with_queue(tomo_viewer):
    task_queue.put((save_and_update_add, (tomo_viewer,)))

def save_and_update_add_2d_with_queue(tomo_viewer):
    task_queue.put((save_and_update_add_2d, (tomo_viewer,)))

def save_and_update_add_6pts_with_queue(tomo_viewer):
    task_queue.put((save_and_update_add_6pts, (tomo_viewer,)))

def save_and_update_delete_with_queue(tomo_viewer):
    task_queue.put((save_and_update_delete, (tomo_viewer,)))


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

    if ccf < 0.2:
        return np.array([0]), None
    
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
            image_data = image_layer.data[:]  # 获取所有选择的点
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
    save_path = tomo_viewer.tomo_path_and_stage.new_label_file_path
    if len(viewer.layers) > 0:
        image_layer = viewer.layers[layer_idx]
        data = np.asarray(image_layer.data).astype(np.float32)
        # data = np.flip(data, axis=1)
        with mrcfile.new(save_path, overwrite=True) as mrc:
            mrc.set_data(data)
            mrc.voxel_size = 17.14
    # show_info('Saved at {}'.format(os.path.abspath(save_path)))

def get_info_from_json(json_file):
    with open(json_file, "r") as f:
        info = json.load(f)
    if not isinstance(info, dict):
        raise ValueError("JSON root must be an object.")
    vesicles = info.get('vesicles', [])
    if not isinstance(vesicles, list):
        raise ValueError("'vesicles' must be a list in JSON.")
    return info

def atomic_write_json(json_file, data):
    json_dir = os.path.dirname(json_file) or '.'
    fd, tmp_path = tempfile.mkstemp(dir=json_dir, prefix='tmp_vesicle_', suffix='.json')
    try:
        with os.fdopen(fd, "w", encoding='utf-8') as out:
            json.dump(data, out)
        os.replace(tmp_path, json_file)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

def parse_vesicle_id(vesicle):
    name = vesicle.get('name', '')
    if not isinstance(name, str):
        return None
    match = VESICLE_NAME_PATTERN.match(name)
    if match is None:
        return None
    return int(match.group(1))

def update_json_file(tomo_viewer, mode, vesicle_to_add=None, delete_label_ids=None):
    json_file = tomo_viewer.tomo_path_and_stage.new_json_file_path
    info = get_info_from_json(json_file)
    vesicles = info.get('vesicles', [])
    result = {
        'removed_count': 0,
        'invalid_name_count': 0,
        'unmatched_ids': [],
    }

    if mode == 'Deleted':
        delete_ids = set(int(i) for i in (delete_label_ids or []) if int(i) > 0)
        if delete_ids:
            filtered_vesicles = []
            removed_ids = set()
            for vesicle in vesicles:
                vesicle_id = parse_vesicle_id(vesicle)
                if vesicle_id is None:
                    result['invalid_name_count'] += 1
                    filtered_vesicles.append(vesicle)
                    continue
                if vesicle_id in delete_ids:
                    removed_ids.add(vesicle_id)
                    result['removed_count'] += 1
                    continue
                filtered_vesicles.append(vesicle)
            vesicles = filtered_vesicles
            result['unmatched_ids'] = sorted(delete_ids - removed_ids)

    elif mode == 'Added':
        vesicles.append(vesicle_to_add)

    info['vesicles'] = vesicles
    atomic_write_json(json_file, info)
    return result

def get_delete_label_ids(label_data, points):
    delete_label_ids = set()
    out_of_bounds_count = 0
    for point in points:
        z, y, x = [int(np.rint(v)) for v in point[:3]]
        if z < 0 or y < 0 or x < 0 or z >= label_data.shape[0] or y >= label_data.shape[1] or x >= label_data.shape[2]:
            out_of_bounds_count += 1
            continue
        label_num = int(label_data[z, y, x])
        if label_num > 0:
            delete_label_ids.add(label_num)
    return sorted(delete_label_ids), out_of_bounds_count

def add_vesicle_show(tomo_viewer, point, add_mode):
    viewer = tomo_viewer.viewer
    ori_tomo = viewer.layers[0].data
    data_max = viewer.layers[LABEL_LAYER_IDX].data.max()
    label_idx = max(data_max, LABEL_START) + 1
    data_to_add, new_added_vesicle = add_vesicle(ori_tomo, point, label_idx, add_mode)
    return data_to_add.astype(np.int16), new_added_vesicle

def delete_picked_vesicle(tomo_viewer, deleted_point):
    viewer = tomo_viewer.viewer
    z, y, x = int(deleted_point[0]), int(deleted_point[1]), int(deleted_point[2])
    # z, y, x = int(deleted_point[0][0]), int(deleted_point[0][1]), int(deleted_point[0][2])
    label_num = viewer.layers[LABEL_LAYER_IDX].data[z, y, x]
    label_layer_data = np.asarray(viewer.layers[LABEL_LAYER_IDX].data)
    label_layer_data[label_layer_data == label_num] = 0
    viewer.layers[LABEL_LAYER_IDX].data = label_layer_data
    viewer.layers[LABEL_LAYER_IDX].refresh()

def add_picked_vesicle(tomo_viewer, data_to_add):
    viewer = tomo_viewer.viewer
    if False: #np.sum(np.sign(viewer.layers[LABEL_LAYER_IDX].data) * np.sign(data_to_add)) > 0:
        tomo_viewer.print('Please reselect two points')
        show_info('Please reselect two points')
    else:
        mask_n = np.sign(data_to_add) * np.sign(viewer.layers[LABEL_LAYER_IDX].data)
        viewer.layers[LABEL_LAYER_IDX].data = viewer.layers[LABEL_LAYER_IDX].data + (1-mask_n) * data_to_add  # update label layer
        viewer.layers[LABEL_LAYER_IDX].refresh()

def save_and_update_delete(tomo_viewer):
    with lock:
        viewer = tomo_viewer.viewer
        if LABEL_LAYER_IDX in viewer.layers:
            if len(viewer.layers[POINT_LAYER_IDX].data) < 1:
                show_info('Please pick a point to delete')
                tomo_viewer.print('Please pick a point to delete')
            else:
                points = save_point_layer(tomo_viewer, POINT_LAYER_IDX, mode='Deleted')
                label_layer = viewer.layers[LABEL_LAYER_IDX]
                label_before = np.asarray(label_layer.data).copy()
                points_backup = np.asarray(points).copy()
                delete_label_ids, out_of_bounds_count = get_delete_label_ids(label_before, points)
                if len(delete_label_ids) == 0:
                    viewer.layers[POINT_LAYER_IDX].data = None
                    msg = 'No valid vesicle labels selected for deletion'
                    if out_of_bounds_count > 0:
                        msg = f'{msg} (out-of-bounds points: {out_of_bounds_count})'
                    show_info(msg)
                    tomo_viewer.print(msg)
                    return

                json_file = tomo_viewer.tomo_path_and_stage.new_json_file_path
                json_before = get_info_from_json(json_file)
                label_after = label_before.copy()
                label_after[np.isin(label_after, delete_label_ids)] = 0

                try:
                    json_result = update_json_file(
                        tomo_viewer,
                        mode='Deleted',
                        vesicle_to_add=None,
                        delete_label_ids=delete_label_ids,
                    )
                    label_layer.data = label_after
                    label_layer.refresh()
                    save_label_layer(tomo_viewer, LABEL_LAYER_IDX)
                    viewer.layers[POINT_LAYER_IDX].data = None

                    summary = (
                        f"Successfully deleted vesicles. points={len(points)}, "
                        f"unique_ids={len(delete_label_ids)}, json_removed={json_result['removed_count']}"
                    )
                    if json_result['invalid_name_count'] > 0:
                        summary = f"{summary}, invalid_json_names={json_result['invalid_name_count']}"
                    if len(json_result['unmatched_ids']) > 0:
                        summary = f"{summary}, unmatched_ids={json_result['unmatched_ids']}"
                    if out_of_bounds_count > 0:
                        summary = f"{summary}, out_of_bounds_points={out_of_bounds_count}"
                    tomo_viewer.print(summary)
                except Exception as exc:
                    rollback_issues = []
                    try:
                        label_layer.data = label_before
                        label_layer.refresh()
                        save_label_layer(tomo_viewer, LABEL_LAYER_IDX)
                    except Exception as rollback_exc:
                        rollback_issues.append(f"label rollback failed: {rollback_exc}")
                    try:
                        atomic_write_json(json_file, json_before)
                    except Exception as rollback_exc:
                        rollback_issues.append(f"json rollback failed: {rollback_exc}")
                    viewer.layers[POINT_LAYER_IDX].data = points_backup
                    err_msg = f'Delete failed and was reverted: {exc}'
                    if rollback_issues:
                        err_msg = f"{err_msg}. {'; '.join(rollback_issues)}"
                    show_info(err_msg)
                    tomo_viewer.print(err_msg)
        else:
            viewer.layers[POINT_LAYER_IDX].data = None
            tomo_viewer.print('Please Make Predict or Start Manual Correction')

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
        save_and_update_delete_with_queue(tomo_viewer)
    del_button = create_button(viewer, 'Delete label (Shortcut: d)', 'delete', 'yellow', 6)
    del_button.clicked.connect(lambda: save_label_image(viewer))

def save_and_update_add(tomo_viewer):
    with lock:
        viewer = tomo_viewer.viewer
        if len(viewer.layers[POINT_LAYER_IDX].data) < 2:
            show_info('Please add two points to define a vesicle')
            tomo_viewer.print('Please add two points to define a vesicle')
        else:
            point = save_point_layer(tomo_viewer, POINT_LAYER_IDX, mode='Added')
            data_to_add, new_added_vesicle = add_vesicle_show(tomo_viewer, point, add_mode='3d')
            if new_added_vesicle is None:
                viewer.layers[POINT_LAYER_IDX].data = None
                tomo_viewer.print('Not a good 3d Vesicle, please reselect')
            else:
                add_picked_vesicle(tomo_viewer, data_to_add)
                viewer.layers[POINT_LAYER_IDX].data = None
                save_label_layer(tomo_viewer, LABEL_LAYER_IDX)
                update_json_file(tomo_viewer, mode='Added', vesicle_to_add=new_added_vesicle[0])
                # tomo_viewer._save_history()
                tomo_viewer.print('Successfully added 3d Vesicle')

def save_and_update_add_2d(tomo_viewer):
    with lock:
        viewer = tomo_viewer.viewer
        if len(viewer.layers[POINT_LAYER_IDX].data) < 2:
            show_info('Please add two points to define a vesicle')
            tomo_viewer.print('Please add two points to define a vesicle')
        else:
            point = save_point_layer(tomo_viewer, POINT_LAYER_IDX, mode='Added')
            data_to_add, new_added_vesicle = add_vesicle_show(tomo_viewer, point, add_mode='2d')
            if new_added_vesicle is None:
                viewer.layers[POINT_LAYER_IDX].data = None
                tomo_viewer.print('Not a good 2d Vesicle, please reselect')
            else:
                add_picked_vesicle(tomo_viewer, data_to_add)
                viewer.layers[POINT_LAYER_IDX].data = None
                save_label_layer(tomo_viewer, LABEL_LAYER_IDX)
                update_json_file(tomo_viewer, mode='Added', vesicle_to_add=new_added_vesicle[0])
                # tomo_viewer._save_history()
                tomo_viewer.print('Successfully added 2d Vesicle')

def save_and_update_add_6pts(tomo_viewer):
    with lock:
        viewer = tomo_viewer.viewer
        if len(viewer.layers[POINT_LAYER_IDX].data) < 6:
            show_info('Please add 6 points to fit a vesicle')
            tomo_viewer.print('Please add 6 points to fit a vesicle')
        else:
            point = save_point_layer(tomo_viewer, POINT_LAYER_IDX, mode='Added_6pts')
            data_to_add, new_added_vesicle = add_vesicle_show(tomo_viewer, point, add_mode='6pts')
            if new_added_vesicle is None:
                viewer.layers[POINT_LAYER_IDX].data = None
                tomo_viewer.print('Not a good 2d Vesicle, please reselect')
            else:
                add_picked_vesicle(tomo_viewer, data_to_add)
                viewer.layers[POINT_LAYER_IDX].data = None
                save_label_layer(tomo_viewer, LABEL_LAYER_IDX)
                update_json_file(tomo_viewer, mode='Added', vesicle_to_add=new_added_vesicle[0])
                # tomo_viewer._save_history()
                tomo_viewer.print('Successfully added 2d Vesicle')

def register_save_shortcut_add(tomo_viewer):
    viewer = tomo_viewer.viewer
    @viewer.bind_key('g', overwrite=True)
    def save_point_image(viewer):
        save_and_update_add_with_queue(tomo_viewer)
    add_button = create_button(viewer, 'Add 3D label (Shortcut: g)', 'add', 'yellow', 0)
    add_button.clicked.connect(lambda: save_point_image(viewer))

def register_save_shortcut_add_2d(tomo_viewer):
    viewer = tomo_viewer.viewer
    @viewer.bind_key('f', overwrite=True)
    def save_point_image(viewer):
        save_and_update_add_2d_with_queue(tomo_viewer)
    add_2d_button = create_button(viewer, 'Add 2D label (Shortcut: f)', 'add', 'white', 1)
    add_2d_button.clicked.connect(lambda: save_point_image(viewer))

def register_save_shortcut_add_6pts(tomo_viewer):
    viewer = tomo_viewer.viewer
    @viewer.bind_key('h', overwrite=True)
    def save_point_image(viewer):
        save_and_update_add_6pts_with_queue(tomo_viewer)
    add_6pts_button = create_button(viewer, 'Add 6pts label (Shortcut: h)', 'polygon_lasso', 'yellow', 2)
    add_6pts_button.clicked.connect(lambda: save_point_image(viewer))

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
