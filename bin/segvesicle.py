#!/usr/bin/env python

import os
# os.environ['QT_API'] = 'pyqt5'
import json
import fire
import napari
from qtpy.QtCore import QTimer, Qt

# from util.check_dependencies import check_dependencies
from ascii_art.print_art import print_byebye_ascii_art, print_pleasewait_ascii_art, print_segvesicle_ascii_art
from folder_list_widget import FolderListWidget
from key_bindings.increment_dims_keys import KeyBinder
from tomo_viewer import TomoViewer
import center_cross

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def add_folder_list_widget(tomo_viewer: TomoViewer):
    folder_list_widget = FolderListWidget(tomo_viewer)
    tomo_viewer.viewer.window.add_dock_widget(folder_list_widget, area='right')
    
# def main(tomo_dir):
def main():
    pid = os.getpid()
    current_path = os.getcwd()

    timer = QTimer()
    timer.setInterval(100)  # 设置定时器间隔，单位为毫秒
    
    viewer = napari.Viewer(title='VesicleSeg')
    # 使用封装类
    
    tomo_viewer = TomoViewer(viewer, current_path, pid)
    
    add_folder_list_widget(tomo_viewer)
    
    key_binder = KeyBinder(timer, tomo_viewer.viewer)
    key_binder.bind_keys()
    
    print_segvesicle_ascii_art()
    
    napari.run()
    
    if tomo_viewer.tomo_path_and_stage.tomo_name is not None:
        if os.path.exists(tomo_viewer.tomo_path_and_stage.new_label_file_path):
            print_pleasewait_ascii_art()
            print(f"Moving {tomo_viewer.tomo_path_and_stage.new_label_file_path} to {tomo_viewer.tomo_path_and_stage.label_path}, please wait...")
            os.system('mv {} {}'.format(tomo_viewer.tomo_path_and_stage.new_label_file_path, tomo_viewer.tomo_path_and_stage.label_path))
        if os.path.exists(tomo_viewer.tomo_path_and_stage.new_json_file_path):
            os.system('mv {} {}'.format(tomo_viewer.tomo_path_and_stage.new_json_file_path, tomo_viewer.tomo_path_and_stage.json_file_path))
            with open(tomo_viewer.tomo_path_and_stage.json_file_path, 'r') as file:
                data = json.load(file)
            # 将JSON数据格式化为多行结构并保存
            with open(tomo_viewer.tomo_path_and_stage.json_file_path, 'w') as file:
                json.dump(data, file, indent=4)
        os.system('rm -r {}'.format(tomo_viewer.tomo_path_and_stage.root_dir))
        print('! '*20)
        print(f"Processed Finished.")
        print('! '*20)
    print_byebye_ascii_art()

if __name__ == '__main__':

    fire.Fire(main)