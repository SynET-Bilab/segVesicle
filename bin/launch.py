#!/usr/bin/env python

import os
import json
import fire
import napari
import mrcfile
import threading
import numpy as np
import pstats
from skimage import exposure

from qtpy import QtCore, QtWidgets
from qtpy.QtCore import QTimer, Qt
from scipy.spatial import KDTree
from skimage.morphology import closing, cube
from napari import Viewer
from napari.settings import get_settings
from napari.resources import ICONS
from napari.utils.notifications import show_info
from napari._qt.widgets.qt_viewer_buttons import QtViewerPushButton

from IsoNet.util.deconvolution import deconv_one
from folder_list_widget import FolderListWidget
from enum import Enum
from three_orthos_viewer import CrossWidget, MultipleViewerWidget
from segVesicle.utils import make_ellipsoid as mk
from morph import density_fit, density_fit_2d, fit_6pts, dis
from global_vars import TOMO_SEGMENTATION_PROGRESS, TomoPath, global_viewer
import center_cross

def add_folder_list_widget(viewer, path, dock_widget):
    folder_list_widget = FolderListWidget(path, dock_widget)
    viewer.window.add_dock_widget(folder_list_widget, area='right')
    
    
# def main(tomo_dir):
def main():
    pid = os.getpid()

    # change increment dims shortcuts
    settings = get_settings()
    settings.shortcuts.shortcuts['napari:increment_dims_left'] = ['PageDown']
    settings.shortcuts.shortcuts['napari:increment_dims_right'] = ['PageUp']

    # 获取当前路径
    current_path = os.getcwd()


    # set default interface
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    main_viewer = global_viewer.window.qt_viewer.parentWidget()
    global dock_widget
    dock_widget = MultipleViewerWidget(global_viewer)
    cross = CrossWidget(global_viewer)
    main_viewer.layout().addWidget(dock_widget)
    global_viewer.window.add_dock_widget(cross, name="Cross", area="left")
    # 将文件夹列表小部件添加到视图中
    add_folder_list_widget(global_viewer, current_path, dock_widget)
    
    napari.run()
    
    from folder_list_widget import tomo_path
    if tomo_path != None:
        os.system('mv {} {}'.format(tomo_path.new_json_file_path, tomo_path.json_file_path))
        os.system('mv {} {}'.format(tomo_path.new_label_file_path, tomo_path.label_path))
        os.system('rm -r {}'.format(tomo_path.root_dir))

if __name__ == '__main__':
    
    # set default params
    LABEL_START = 10000  # large enough to avoid overlap with original label
    LABEL_LAYER_IDX = 'label'
    POINT_LAYER_IDX = 'edit vesicles'
    ORI_LAYER_IDX = 'ori_tomo'
    NUM_POINT = 0
    global added_vesicle_num
    added_vesicle_num = 0
    label_history = None

    fire.Fire(main)
