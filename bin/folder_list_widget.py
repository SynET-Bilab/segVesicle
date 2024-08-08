import os
import json
import re
import numpy as np
import SimpleITK as sitk
import mrcfile

from qtpy.QtWidgets import QProgressDialog, QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QCheckBox, QPushButton, QFileDialog
from qtpy.QtCore import Qt

from napari.utils.notifications import show_info
from key_bindings.add_del_label import add_button_and_register_add_and_delete

from tomo_viewer import TomoViewer
from global_vars import TOMO_NAME
from util.add_layer_with_right_contrast import add_layer_with_right_contrast
from util.resample import resample_image

LABEL_START = 10000  # large enough to avoid overlap with original labe
LABEL_LAYER_IDX = 'label'
POINT_LAYER_IDX = 'edit vesicles'
# ORI_LAYER_IDX = 'ori_tomo'
NUM_POINT = 0
global added_vesicle_num
added_vesicle_num = 0
label_history = None
tomo_path = None

def get_tomo(path):
    with mrcfile.open(path) as mrc:
        data = mrc.data
    # data = np.flip(data, axis=1)
    return data

        
class FolderListWidget(QWidget):
    def __init__(self, tomo_viewer: TomoViewer):
        super().__init__()
        self.tomo_viewer = tomo_viewer
        self.path = self.tomo_viewer.tomo_path_and_stage.current_path  # 从tomo_viewer获取路径
        self.tomo_path = None
        # 保存路径为实例变量
        self.layout = QVBoxLayout()
        
        # 添加打开文件夹按钮
        self.open_folder_button = QPushButton("Open Folder")
        self.open_folder_button.clicked.connect(self.open_folder_dialog)
        self.layout.addWidget(self.open_folder_button)
        
        # 添加创建segVesicle.batch按钮
        self.create_segvesicle_batch_button = QPushButton("Create segVesicle.batch")
        self.create_segvesicle_batch_button.clicked.connect(self.create_segvesicle_batch)
        self.layout.addWidget(self.create_segvesicle_batch_button)
        
        self.list_widget = QListWidget()
        self.layout.addWidget(self.list_widget)
        self.setLayout(self.layout)
        
        self.state_file = os.path.join(self.path, 'segVesicle_QCheckBox_state.json')
        self.checkbox_states = self.load_checkbox_states()
        
        self.populate_list(self.path)
        
        self.dock_widget = self.tomo_viewer.multiple_viewer_widget
        self.tomo_viewer.print("Welcome to the Vesicle Segmentation Software, version 0.1.")
        self.tomo_viewer.print("For instructions and keyboard shortcuts, please refer to the help documentation available in the '?' section at the top right corner.")
        try:
            import tensorflow as tf
        except ImportError:
            self.tomo_viewer.print("TensorFlow with GPU support is not installed. Please download and install it to use the IsoNet correction and vesicle prediction model.")
 

    def open_folder_dialog(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", os.getcwd())
        # 清除上一个文件夹的缓存
        if self.tomo_path != None:
            os.system('mv {} {}'.format(self.tomo_path.new_json_file_path, self.tomo_path.json_file_path))
            with open(self.tomo_path.json_file_path, 'r') as file:
                data = json.load(file)
            # 将JSON数据格式化为多行结构并保存
            with open(self.tomo_path.json_file_path, 'w') as file:
                json.dump(data, file, indent=4)
            os.system('mv {} {}'.format(self.tomo_path.new_label_file_path, self.tomo_path.label_path))
            os.system('rm -r {}'.format(self.tomo_path.root_dir))
            message = f"Saved tomo {self.tomo_viewer.tomo_path_and_stage.tomo_name}."
            self.tomo_viewer.print(message)
        if folder_path:
            self.path = folder_path
            self.tomo_viewer.tomo_path_and_stage.current_path = folder_path
            self.state_file = os.path.join(self.path, 'segVesicle_QCheckBox_state.json')
            self.checkbox_states = self.load_checkbox_states()
            self.list_widget.clear()
            self.populate_list(folder_path)
            message = f"Current path changed to: {self.path}"
            self.tomo_viewer.print(message)

    def create_segvesicle_batch(self):
        current_path = self.tomo_viewer.tomo_path_and_stage.current_path
        import subprocess
        if current_path:
            files = []
            for root, dirs, filenames in os.walk(current_path):
                for filename in filenames:
                    if filename.endswith('.rec') or filename.endswith('.mrc'):
                        dir_path = os.path.relpath(root, current_path)
                        if dir_path not in [os.path.dirname(f) for f in files]:
                            files.append(os.path.join(dir_path, filename))
            
            with open(os.path.join(current_path, 'segVesicle.batch'), 'w') as f:
                for file in files:
                    f.write(file + '\n')
            self.state_file = os.path.join(self.path, 'segVesicle_QCheckBox_state.json')
            self.checkbox_states = self.load_checkbox_states()
            self.list_widget.clear()
            self.populate_list(self.tomo_viewer.tomo_path_and_stage.current_path)
            

    def load_checkbox_states(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {}

    def save_checkbox_states(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.checkbox_states, f)

    def populate_list(self, path):
        # 自定义排序函数
        def sort_key(item):
            match = re.match(r"(pp?)(\d+)", item)
            if match:
                prefix, number = match.groups()
                return (prefix, int(number))
            return (item, 0)
        
        batch_file_path = os.path.join(path, 'segVesicle.batch')
        if os.path.exists(batch_file_path):
            with open(batch_file_path, 'r') as file:
                lines = file.readlines()
            
            folders = sorted(
                {line.strip().split('/')[0] for line in lines},
                key=sort_key
            )
            
            for item in folders:
                list_item = QListWidgetItem("        " + item)
                self.list_widget.addItem(list_item)
                checkbox = QCheckBox()
                
                # 设置 QCheckBox 的状态
                if item in self.checkbox_states:
                    checkbox.setChecked(self.checkbox_states[item])
                
                # 当状态改变时更新状态字典并保存到文件
                checkbox.stateChanged.connect(lambda state, item=item: self.update_checkbox_state(state, item))
                self.list_widget.setItemWidget(list_item, checkbox)

            # 先断开之前的绑定
            try:
                self.list_widget.itemDoubleClicked.disconnect()
            except TypeError:
                pass

            self.list_widget.itemDoubleClicked.connect(self.on_item_double_click)
        else:
            pass

    def update_checkbox_state(self, state, item):
        self.checkbox_states[item] = (state == 2)
        self.save_checkbox_states()

    def save_tomo_files(self):
        tomo_viewer = self.tomo_viewer
        tomo_path = self.tomo_path
        all_paths_exist = True

        # 判断并执行路径相关操作
        if os.path.exists(tomo_path.new_json_file_path):
            os.system('mv {} {}'.format(tomo_path.new_json_file_path, tomo_path.json_file_path))
            # tomo_viewer.print(f"{tomo_path.new_json_file_path} moved to {tomo_path.json_file_path}")
        else:
            tomo_viewer.print(f"Path {tomo_path.new_json_file_path} does not exist")
            all_paths_exist = False

        if os.path.exists(tomo_path.json_file_path):
            with open(tomo_path.json_file_path, 'r') as file:
                data = json.load(file)
            # 将JSON数据格式化为多行结构并保存
            with open(tomo_path.json_file_path, 'w') as file:
                json.dump(data, file, indent=4)
        else:
            # tomo_viewer.print(f"Path {tomo_path.json_file_path} does not exist")
            all_paths_exist = False

        if os.path.exists(tomo_path.new_label_file_path):
            os.system('mv {} {}'.format(tomo_path.new_label_file_path, tomo_path.label_path))
            # tomo_viewer.print(f"{tomo_path.new_label_file_path} moved to {tomo_path.label_path}")
        else:
            tomo_viewer.print(f"Path {tomo_path.new_label_file_path} does not exist")
            all_paths_exist = False

        if os.path.exists(tomo_path.root_dir):
            os.system('rm -r {}'.format(tomo_path.root_dir))
            # tomo_viewer.print(f"{tomo_path.root_dir} removed")
        else:
            tomo_viewer.print(f"Path {tomo_path.root_dir} does not exist")
            all_paths_exist = False

        # 打印保存信息
        if all_paths_exist:
            message = f"Saved tomo {tomo_viewer.tomo_path_and_stage.tomo_name}."
            tomo_viewer.print(message)
        else:
            tomo_viewer.print("Save failed. One or more paths do not exist.")

    def on_item_double_click(self, item):
        self.progress_dialog = QProgressDialog("Processing...", 'Cancel', 0, 100, self)
        self.progress_dialog.setWindowTitle('Opening')
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setValue(0)
        self.progress_dialog.show()
        
        # 清除上一个文件夹的缓存
        if self.tomo_path != None:
            self.save_tomo_files()
        
        # # 清除之前的层
        def remove_all_layers(viewer):
            layer_names = [layer.name for layer in viewer.layers]
            for name in layer_names:
                viewer.layers.remove(name)
        remove_all_layers(self.tomo_viewer.viewer)
        
        item_name = item.text().strip()
        
        self.progress_dialog.setValue(20)

        self.tomo_viewer.set_tomo_name(item_name)
        TOMO_NAME = item_name
        root_dir = self.tomo_viewer.tomo_path_and_stage.root_dir
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        self.tomo_path = self.tomo_viewer.tomo_path_and_stage
        
        if self.tomo_path.progress_stage.name == 'MANUALLY_CORRECT_LABELS':
            os.system('cp {} {}'.format(self.tomo_path.json_file_path, self.tomo_path.new_json_file_path))
            if not os.path.exists(self.tomo_path.ori_label_path):
                os.system('cp {} {}'.format(self.tomo_path.label_path, self.tomo_path.ori_label_path))
                os.system('cp {} {}'.format(self.tomo_path.json_file_path, self.tomo_path.ori_json_file_path))
            
            if os.path.exists(self.tomo_path.isonet_tomo_path):
                tomo = get_tomo(self.tomo_path.isonet_tomo_path)
                add_layer_with_right_contrast(tomo, 'corrected_tomo', self.tomo_viewer.viewer)
            else:
                def choose_tomo():
                # Create the dialog
                    from qtpy.QtWidgets import QFileDialog, QDialog, QVBoxLayout, QPushButton, QLineEdit, QLabel, QHBoxLayout
                    dialog = QDialog(self.tomo_viewer.main_viewer)
                    dialog.setWindowTitle('Open Tomogram')

                    layout = QVBoxLayout()

                    # File selection button
                    file_select_layout = QHBoxLayout()
                    file_label = QLabel('File:')
                    initial_path = os.path.join(self.tomo_viewer.tomo_path_and_stage.current_path, self.tomo_viewer.tomo_path_and_stage.tomo_name)
                    file_line_edit = QLineEdit(initial_path)
                    file_select_button = QPushButton('Select File')
                    file_select_layout.addWidget(file_label)
                    file_select_layout.addWidget(file_line_edit)
                    file_select_layout.addWidget(file_select_button)
                    layout.addLayout(file_select_layout)

                    def select_file():
                        options = QFileDialog.Options()
                        file_path, _ = QFileDialog.getOpenFileName(
                            dialog,
                            "Select File",
                            initial_path,
                            "MRC or REC Files (*.mrc *.rec);;All Files (*)",
                            options=options
                        )
                        if file_path:
                            file_line_edit.setText(file_path)

                    file_select_button.clicked.connect(select_file)

                    # Pixel size input
                    pixel_size_layout = QHBoxLayout()
                    pixel_size_label = QLabel('Pixel Size:')
                    pixel_size_input = QLineEdit('17.142')
                    pixel_size_layout.addWidget(pixel_size_label)
                    pixel_size_layout.addWidget(pixel_size_input)
                    layout.addLayout(pixel_size_layout)

                    # Apply button
                    apply_button = QPushButton('Apply')
                    layout.addWidget(apply_button)

                    def apply_resample_image():
                        file_path = file_line_edit.text()
                        try:
                            pixel_size = float(pixel_size_input.text())
                        except ValueError:
                            self.tomo_viewer.print("Invalid pixel size.")
                            return

                        data = resample_image(file_path, pixel_size)
                        data = sitk.GetArrayFromImage(data)
                        add_layer_with_right_contrast(data, 'tomo', self.tomo_viewer.viewer)
                            
                        message = f"Successfully opened tomo {file_path}."
                        self.tomo_viewer.print(message)
                        dialog.accept()
                    
                    try:
                        apply_button.clicked.disconnect(apply_resample_image)
                    except TypeError:
                        # 如果未绑定，则会抛出 TypeError 异常
                        pass
                    # 重新绑定
                    apply_button.clicked.connect(apply_resample_image)

                    dialog.setLayout(layout)
                    dialog.exec_()
                choose_tomo()
            
            self.progress_dialog.setValue(40)
            
            self.progress_dialog.setValue(60)
            self.tomo_viewer.viewer.add_labels(get_tomo(self.tomo_path.label_path).astype(np.int16), name='label')  # add label layer
            self.progress_dialog.setValue(80)
            self.tomo_viewer.viewer.add_points(name='edit vesicles', ndim=3, size=4)  # add an empty Points layer
            self.progress_dialog.setValue(90)
        
            self.tomo_viewer.viewer.layers['label'].opacity = 0.5
            self.tomo_viewer.viewer.layers['edit vesicles'].mode = 'ADD'
            
            self.dock_widget.viewer_model1.camera.zoom = 0.9
            self.dock_widget.viewer_model2.camera.zoom = 0.9
            # self.dock_widget.viewer_model3.camera.zoom = 2
            
            add_button_and_register_add_and_delete(self.tomo_viewer)
            self.tomo_viewer.register()
            
            message = f"Successfully opened tomo {self.tomo_viewer.tomo_path_and_stage.tomo_name}."
            self.tomo_viewer.print(message)
            self.tomo_viewer.show_current_state()
            
            self.progress_dialog.setValue(100)
            self.progress_dialog.close()
        elif self.tomo_path.progress_stage.name == 'OPEN_TOMO':
            
            add_button_and_register_add_and_delete(self.tomo_viewer)
            self.tomo_viewer.register()
            self.tomo_viewer.show_current_state()

            self.progress_dialog.close()
            
        elif self.tomo_path.progress_stage.name == 'MAKE_DECONVOLUTION':
            tomo = get_tomo(self.tomo_path.ori_tomo_path)
            
            add_layer_with_right_contrast(tomo, 'ori_tomo', self.tomo_viewer.viewer)
            
            
            self.tomo_viewer.viewer.add_points(name='edit vesicles', ndim=3, size=4)
            self.tomo_viewer.viewer.layers['edit vesicles'].mode = 'ADD'

            add_button_and_register_add_and_delete(self.tomo_viewer)
            self.tomo_viewer.register()
            self.tomo_viewer.show_current_state()
            
            self.dock_widget.viewer_model1.camera.zoom = 0.9
            self.dock_widget.viewer_model2.camera.zoom = 0.9
            self.progress_dialog.close()
            
        elif self.tomo_path.progress_stage.name == 'MAKE_CORRECTION':
            tomo = get_tomo(self.tomo_path.deconv_tomo_path)
            
            add_layer_with_right_contrast(tomo, 'deconv_tomo', self.tomo_viewer.viewer)
            
            
            self.tomo_viewer.viewer.add_points(name='edit vesicles', ndim=3, size=4)
            self.tomo_viewer.viewer.layers['edit vesicles'].mode = 'ADD'

            add_button_and_register_add_and_delete(self.tomo_viewer)
            self.tomo_viewer.register()
            self.tomo_viewer.show_current_state()
            
            self.dock_widget.viewer_model1.camera.zoom = 0.9
            self.dock_widget.viewer_model2.camera.zoom = 0.9
            self.progress_dialog.close()
            
        elif self.tomo_path.progress_stage.name == 'MAKE_PREDICT':
            tomo = get_tomo(self.tomo_path.isonet_tomo_path)
            add_layer_with_right_contrast(tomo, 'corrected_tomo', self.tomo_viewer.viewer)
            
            tomo = get_tomo(self.tomo_path.deconv_tomo_path)
            add_layer_with_right_contrast(tomo, 'deconv_tomo', self.tomo_viewer.viewer)
            self.tomo_viewer.viewer.layers['deconv_tomo'].visible = False
            
            self.tomo_viewer.viewer.add_points(name='edit vesicles', ndim=3, size=4)
            self.tomo_viewer.viewer.layers['edit vesicles'].mode = 'ADD'

            add_button_and_register_add_and_delete(self.tomo_viewer)
            self.tomo_viewer.register()
            self.tomo_viewer.show_current_state()
            
            self.dock_widget.viewer_model1.camera.zoom = 0.9
            self.dock_widget.viewer_model2.camera.zoom = 0.9
            self.progress_dialog.close()