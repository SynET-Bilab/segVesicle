import os
import json
import sys
import re
import numpy as np
import mrcfile
import threading

from scipy.spatial import KDTree
from qtpy.QtWidgets import QProgressDialog, QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QCheckBox, QHBoxLayout, QLabel, QApplication
from qtpy.QtCore import Qt
from skimage.morphology import closing, cube
from napari import Viewer
from napari.resources import ICONS
from napari.utils.notifications import show_info
from napari._qt.widgets.qt_viewer_buttons import QtViewerPushButton
from napari.qt.threading import thread_worker

from global_vars import TomoPath, global_viewer
from enum import Enum
from segVesicle.utils import make_ellipsoid as mk
from morph import density_fit, density_fit_2d, fit_6pts, dis
from global_vars import TOMO_SEGMENTATION_PROGRESS, TomoPath, global_viewer

LABEL_START = 10000  # large enough to avoid overlap with original labe
LABEL_LAYER_IDX = 'label'
POINT_LAYER_IDX = 'edit vesicles'
ORI_LAYER_IDX = 'ori_tomo'
NUM_POINT = 0
global added_vesicle_num
added_vesicle_num = 0
label_history = None
tomo_path = None

# class LabelHistory:
#     def __init__(self, layer):
#         self.layer = layer
#         self.history = []
#         self.index = -1
#         self.max_history = 10  # 可以根据需要调整历史记录的最大数量

#     def save_state(self):
#         if len(self.history) >= self.max_history:
#             self.history.pop(0)
#         self.history.append(self.layer.data.copy())
#         self.index = len(self.history) - 1

#     def undo(self):
#         if self.index > 0:
#             self.index -= 1
#             self.layer.data = self.history[self.index]

#     def redo(self):
#         if self.index < len(self.history) - 1:
#             self.index += 1
#             self.layer.data = self.history[self.index]

def print_in_widget(message):
    pass
    # if dock_widget:
    #     dock_widget.message_signal.emit(message)

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
        # global label_history
        # label_history.save_state()


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
    del_button = create_button(viewer, 'Delete label (Shortcut: d)', 'delete', 'yellow', 6)
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
        # global label_history
        # label_history.save_state()


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
        # global label_history
        # label_history.save_state()
        
        
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
        # global label_history
        # label_history.save_state()


def register_save_shortcut_add(viewer, root_dir, new_json_file_path):
    @viewer.bind_key('g', overwrite=True)
    def save_point_image(viewer):
        threading.Thread(target=save_and_update_add, args=(viewer, root_dir, new_json_file_path)).start()
    # 创建添加按钮并将其点击事件与 save_point_image 函数绑定
    add_button = create_button(viewer, 'Ddd 3D label (Shortcut: g)', 'add', 'yellow', 3)
    add_button.clicked.connect(lambda: save_point_image(viewer))
    
    
def register_save_shortcut_add_2d(viewer, root_dir, new_json_file_path):
    @viewer.bind_key('f', overwrite=True)
    def save_point_image(viewer):
        threading.Thread(target=save_and_update_add_2d, args=(viewer, root_dir, new_json_file_path)).start()
    # 创建添加 2D 按钮并将其点击事件与 save_point_image 函数绑定
    add_2d_button = create_button(viewer, 'Ddd 2D label (Shortcut: f)', 'add', 'white', 4)
    add_2d_button.clicked.connect(lambda: save_point_image(viewer))


def register_save_shortcut_add_6pts(viewer, root_dir, new_json_file_path):
    @viewer.bind_key('p', overwrite=True)
    def save_point_image(viewer):
        threading.Thread(target=save_and_update_add_6pts, args=(viewer, root_dir, new_json_file_path)).start()
    # 创建添加 6pts 按钮并将其点击事件与 save_and_update_add_6pts 函数绑定
    add_6pts_button = create_button(viewer, 'add 6pts label (Shortcut: p)', 'polygon_lasso', 'yellow', 5)
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

def add_folder_list_widget(viewer, path):
    folder_list_widget = FolderListWidget(path)
    viewer.window.add_dock_widget(folder_list_widget, area='right')


def add_button_and_register_add_and_delete(viewer: Viewer, root_dir, new_json_file_path):
    layer_buttons = viewer.window.qt_viewer.layerButtons
    # 获取layer_buttons的布局
    layout = layer_buttons.layout()

    # 通过遍历布局中的所有项目来删除所有按钮
    while layout.count():
        item = layout.takeAt(0)  # 每次都从布局中取出第一个项目
        if item is not None:
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()  # 删除小部件
    
    register_save_shortcut_add(viewer, root_dir, new_json_file_path)
    register_save_shortcut_add_2d(viewer, root_dir, new_json_file_path)
    register_save_shortcut_add_6pts(viewer, root_dir, new_json_file_path)
    register_save_shortcut_delete(viewer, root_dir, new_json_file_path)
    # register_shortcut_crop_image(viewer)
    
    # layer_buttons = viewer.window.qt_viewer.layerButtons

    # # 删除位置在1，2，3的按钮
    # for i in [4, 2, 1, 0]:
    #     item = layer_buttons.layout().takeAt(i)
    #     if item is not None:
    #         widget = item.widget()
    #         if widget is not None:
    #             widget.deleteLater()

class FolderListWidget(QWidget):
    def __init__(self, path):
        super().__init__()
        self.path = path  # 保存路径为实例变量
        self.layout = QVBoxLayout()
        self.list_widget = QListWidget()
        self.layout.addWidget(self.list_widget)
        self.setLayout(self.layout)
        
        self.populate_list(path)

    def populate_list(self, path):
        # 自定义排序函数
        def sort_key(item):
            match = re.match(r"(pp?)(\d+)", item)
            if match:
                prefix, number = match.groups()
                return (prefix, int(number))
            return (item, 0)
        # 获取所有以 "pp" 开头的文件夹，并按名称排序
        folders = sorted(
            [item for item in os.listdir(path) 
            if os.path.isdir(os.path.join(path, item)) 
            and item.startswith("p")],
            key=sort_key
        )
        
        for item in folders:
            list_item = QListWidgetItem("        " + item)
            self.list_widget.addItem(list_item)
            checkbox = QCheckBox()
            self.list_widget.setItemWidget(list_item, checkbox)

        self.list_widget.itemDoubleClicked.connect(self.on_item_double_click)

    def on_item_double_click(self, item):
        self.progress_dialog = QProgressDialog("Processing...", 'Cancel', 0, 100, self)
        self.progress_dialog.setWindowTitle('Opening')
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setValue(0)
        self.progress_dialog.show()

        # global label_history
        # label_history = None
        
        # 清除上一个文件夹的缓存
        global tomo_path
        if tomo_path != None:
            os.system('mv {} {}'.format(tomo_path.new_json_file_path, tomo_path.json_file_path))
            os.system('mv {} {}'.format(tomo_path.new_label_file_path, tomo_path.label_path))
            os.system('rm -r {}'.format(tomo_path.root_dir))
        # 清除之前的层
        def remove_layer_if_exists(viewer, layer_name):
            if layer_name in viewer.layers:
                viewer.layers.remove(layer_name)
        layer_names = ['label', 'corrected_tomo', 'edit vesicles']
        for name in layer_names:
            remove_layer_if_exists(global_viewer, name)
        
        pid = os.getpid()
        item_name = item.text().strip()
        root_dir = os.path.abspath(item_name) + '/ves_seg/temp/'
        self.progress_dialog.setValue(20)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        tomo_path=TomoPath(item_name, root_dir, pid)
        
        self.progress_dialog.setValue(30)
        os.system('cp {} {}'.format(tomo_path.json_file_path, tomo_path.new_json_file_path))
        if not os.path.exists(tomo_path.ori_label_path):
            os.system('cp {} {}'.format(tomo_path.label_path, tomo_path.ori_label_path))
            os.system('cp {} {}'.format(tomo_path.json_file_path, tomo_path.ori_json_file_path))
        
        lambda_scale = 0.35
        tomo = get_tomo(tomo_path.isonet_tomo_path)
        mi, ma = (tomo.max() - tomo.min()) * lambda_scale + tomo.min(), tomo.max() - (tomo.max() - tomo.min()) * lambda_scale
        
        self.progress_dialog.setValue(40)
        # global global_viewer
        global_viewer.add_labels(get_tomo(tomo_path.label_path).astype(np.int16), name='label')  # add label layer
        self.progress_dialog.setValue(60)
        global_viewer.add_image(get_tomo(tomo_path.isonet_tomo_path), name='corrected_tomo')  # add isonet treated tomogram layer
        self.progress_dialog.setValue(80)
        global_viewer.add_points(name='edit vesicles', ndim=3, size=4)  # add an empty Points layer
        self.progress_dialog.setValue(90)
        # label_history = LabelHistory(label_layer)
        # label_history.save_state()
    
        # # 监听键盘事件，实现撤销和重做操作
        # @global_viewer.bind_key('Control-z')
        # def undo(viewer):
        #     label_history.undo()

        # @global_viewer.bind_key('Control-Shift-z')
        # def redo(viewer):
        #     label_history.redo()
    
        global_viewer.layers['corrected_tomo'].opacity = 0.5
        global_viewer.layers['corrected_tomo'].contrast_limits = [mi, ma]
        global_viewer.layers['edit vesicles'].mode = 'ADD'
        
        add_button_and_register_add_and_delete(global_viewer, root_dir, tomo_path.new_json_file_path)
        self.progress_dialog.setValue(100)
        self.progress_dialog.close()