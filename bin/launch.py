#!/usr/bin/env python

import os


# os.environ['QT_API'] = 'pyqt5'
import json
import fire
import napari


from qtpy.QtCore import QTimer, Qt


from folder_list_widget import FolderListWidget
from key_bindings.increment_dims_keys import KeyBinder
from tomo_viewer import TomoViewer
import center_cross

def add_folder_list_widget(tomo_viewer: TomoViewer):
    folder_list_widget = FolderListWidget(tomo_viewer)
    tomo_viewer.viewer.window.add_dock_widget(folder_list_widget, area='right')
    
    
# def main(tomo_dir):
def main():
    pid = os.getpid()

    # 获取当前路径
    current_path = os.getcwd()

    timer = QTimer()
    timer.setInterval(100)  # 设置定时器间隔，单位为毫秒
    
    viewer = napari.Viewer()
    # 使用封装类
    tomo_viewer = TomoViewer(viewer, current_path, pid)
    
    add_folder_list_widget(tomo_viewer)
    
    key_binder = KeyBinder(timer, tomo_viewer.viewer)
    key_binder.bind_keys()
    
    napari.run()
    
    if tomo_viewer.tomo_path_and_stage.tomo_name != None:
        os.system('mv {} {}'.format(tomo_viewer.tomo_path_and_stage.new_json_file_path, tomo_viewer.tomo_path_and_stage.json_file_path))
        with open(tomo_viewer.tomo_path_and_stage.json_file_path, 'r') as file:
            data = json.load(file)
        # 将JSON数据格式化为多行结构并保存
        with open(tomo_viewer.tomo_path_and_stage.json_file_path, 'w') as file:
            json.dump(data, file, indent=4)
        os.system('mv {} {}'.format(tomo_viewer.tomo_path_and_stage.new_label_file_path, tomo_viewer.tomo_path_and_stage.label_path))
        os.system('rm -r {}'.format(tomo_viewer.tomo_path_and_stage.root_dir))

if __name__ == '__main__':

    fire.Fire(main)
