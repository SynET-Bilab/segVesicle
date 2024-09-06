import numpy as np
import mrcfile
import napari
import os
import subprocess
import json
import tempfile

from qtpy import QtCore, QtWidgets
from napari.utils.notifications import show_info
import pandas as pd
from pathlib import Path

from three_orthos_viewer import CrossWidget, MultipleViewerWidget
from tomo_path_and_stage import TomoPathAndStage
from qtpy.QtWidgets import QFileDialog, QDialog, QVBoxLayout, QPushButton, QLineEdit, QLabel, QHBoxLayout, QMessageBox

from collections import deque

from window.deconv_window import DeconvWindow
from window.correction_window import CorrectionWindow
from util.add_layer_with_right_contrast import add_layer_with_right_contrast
from util.predict_vesicle import predict_label, morph_process, vesicle_measure, vesicle_rendering
from util.resample import resample_image
from widget.function_widget import ToolbarWidget

class TomoViewer:
    def __init__(self, viewer: napari.Viewer, current_path: str, pid: int):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
        self.viewer: napari.Viewer = viewer
        self.toolbar_widget = ToolbarWidget()
        self.main_viewer = self.viewer.window.qt_viewer.parentWidget()
        self.multiple_viewer_widget: MultipleViewerWidget = MultipleViewerWidget(self.viewer)
        self.tomo_path_and_stage: TomoPathAndStage = TomoPathAndStage(current_path, pid)
        self.cross_widget: CrossWidget = CrossWidget(self.viewer)
        self.main_viewer.layout().addWidget(self.multiple_viewer_widget)
        self.viewer.window.add_dock_widget(self.cross_widget, name="Cross", area="left")
        self.viewer.window.add_dock_widget(self.toolbar_widget, area='left', name='Tools')
        self.show_current_state()
        
    def set_tomo_name(self, tomo_name: str):
        self.tomo_path_and_stage.set_tomo_name(tomo_name)
        
    def show_current_state(self):
        progress_value = self.tomo_path_and_stage.determine_progress()
        # progress_value = self.tomo_path_and_stage.progress_stage.value
        # 将进度状态展示在QLineEdit里
        self.multiple_viewer_widget.utils_widget.ui.progressStage.setText(progress_value.value)
    
    def print(self, message):
        self.multiple_viewer_widget.print_in_widget(message)
        
    def register(self):
        self.register_correction_tomo()
        self.register_deconv_tomo()
        self.register_open_ori_tomo()
        self.register_draw_area_mod()
        self.register_manualy_correction()
        try:
            self.toolbar_widget.finish_isonet_button.clicked.disconnect()
        except TypeError:
            pass
        self.toolbar_widget.finish_isonet_button.clicked.connect(self.on_finish_isonet_clicked)
        
        try:
            self.toolbar_widget.predict_button.clicked.disconnect()
        except TypeError:
            pass
        self.toolbar_widget.predict_button.clicked.connect(self.predict_clicked)
        
        try:
            self.toolbar_widget.draw_memb_button.clicked.disconnect()
        except TypeError:
            pass
        self.toolbar_widget.draw_memb_button.clicked.connect(self.register_draw_memb_mod)
        
        try:
            self.toolbar_widget.visualize_button.clicked.disconnect()
        except TypeError:
            pass
        self.toolbar_widget.visualize_button.clicked.connect(self.register_vis_memb)
        
        try:
            self.toolbar_widget.stsyseg_button.clicked.disconnect()
        except TypeError:
            pass
        self.toolbar_widget.stsyseg_button.clicked.connect(self.register_seg_memb)
        
    def register_open_ori_tomo(self):
        def button_clicked():
        # Create the dialog
            dialog = QDialog(self.main_viewer)
            dialog.setWindowTitle('Open Original Tomogram')

            layout = QVBoxLayout()

            # File selection button
            file_select_layout = QHBoxLayout()
            file_label = QLabel('File:')
            initial_path = os.path.join(self.tomo_path_and_stage.current_path, self.tomo_path_and_stage.tomo_name)
            self.file_line_edit = QLineEdit(initial_path)
            file_select_button = QPushButton('Select File')
            file_select_layout.addWidget(file_label)
            file_select_layout.addWidget(self.file_line_edit)
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
                    self.file_line_edit.setText(file_path)

            file_select_button.clicked.connect(select_file)

            # Pixel size input
            pixel_size_layout = QHBoxLayout()
            pixel_size_label = QLabel('Pixel Size:')
            self.pixel_size_input = QLineEdit('17.142')
            pixel_size_layout.addWidget(pixel_size_label)
            pixel_size_layout.addWidget(self.pixel_size_input)
            layout.addLayout(pixel_size_layout)

            # Apply button
            apply_button = QPushButton('Apply')
            layout.addWidget(apply_button)

            def apply_resample_image():
                file_path = self.file_line_edit.text()
                try:
                    pixel_size = float(self.pixel_size_input.text())
                except ValueError:
                    self.print("Invalid pixel size.")
                    return

                # Show progress dialog
                from qtpy.QtWidgets import QProgressDialog
                from qtpy.QtCore import Qt
                self.progress_dialog = QProgressDialog("Processing...", 'Cancel', 0, 100, self.main_viewer)
                self.progress_dialog.setWindowTitle('Opening')
                self.progress_dialog.setWindowModality(Qt.WindowModal)
                self.progress_dialog.setValue(0)
                self.progress_dialog.show()

                # Processing
                self.progress_dialog.setValue(50)
                data = resample_image(file_path, pixel_size)
                if 'ori_tomo' in self.viewer.layers:
                    self.viewer.layers.remove('ori_tomo')
                add_layer_with_right_contrast(data, 'ori_tomo', self.viewer)
                rec_folder_path = os.path.dirname(self.tomo_path_and_stage.rec_tomo_path)
                if not os.path.exists(rec_folder_path):
                    os.makedirs(rec_folder_path)
                with mrcfile.new(self.tomo_path_and_stage.rec_tomo_path, overwrite=True) as output_mrc:
                    output_mrc.set_data(data)
                    output_mrc.voxel_size = 17.14
                if 'corrected_tomo' in self.viewer.layers:
                    self.viewer.layers['corrected_tomo'].visible = False
                if 'edit vesicles' in self.viewer.layers:
                    self.viewer.layers.selection.active = self.viewer.layers['edit vesicles']
                
                self.viewer.layers.move(self.viewer.layers.index(self.viewer.layers['ori_tomo']), 0)
                
                if 'edit vesicles' in self.viewer.layers:
                    self.viewer.layers.remove('edit vesicles')
                self.viewer.add_points(name='edit vesicles', ndim=3, size=4)
                self.viewer.layers['edit vesicles'].mode = 'ADD'
                self.multiple_viewer_widget.viewer_model1.camera.zoom = 0.9
                self.multiple_viewer_widget.viewer_model2.camera.zoom = 0.9
                self.progress_dialog.setValue(100)
                message = f"Successfully opened the original image {file_path}."
                self.print(message)
                self.show_current_state()
                dialog.accept()

            apply_button.clicked.connect(apply_resample_image)
            dialog.setLayout(layout)
            dialog.exec_()

        try:
            self.toolbar_widget.open_ori_image_button.clicked.disconnect()
        except TypeError:
            pass

        self.toolbar_widget.open_ori_image_button.clicked.connect(button_clicked)
        
    def register_deconv_tomo(self):
        def open_deconv_window():
            if 'ori_tomo' in self.viewer.layers:
                if len(self.viewer.layers['edit vesicles'].data) == 2:
                    self.deconv_window = DeconvWindow(self)
                    self.deconv_window.show()
                else:
                    self.viewer.layers['edit vesicles'].data = None
                    self.print('Please add two points to define deconvolution area.')
                    show_info('Please add two points to define deconvolution area.')
            else:
                self.print('Please open original tomo.')
                show_info('Please open original tomo.')
        try:
            self.toolbar_widget.deconvolution_button.clicked.disconnect()
        except TypeError:
            pass
        self.toolbar_widget.deconvolution_button.clicked.connect(open_deconv_window)
        
    def register_correction_tomo(self):
        def open_correction_window():
            if 'deconv_tomo' in self.viewer.layers:
                self.correction_window = CorrectionWindow(self)
                self.correction_window.show()
            else:
                self.print('Please perform deconvolution.')
                show_info('Please perform deconvolution.')
        try:
            self.toolbar_widget.correction_button.clicked.disconnect()
        except TypeError:
            pass
        
        self.toolbar_widget.correction_button.clicked.connect(open_correction_window)
        
    def on_finish_isonet_clicked(self):
        self.toolbar_widget.tabs.setCurrentIndex(1)
        
    def predict_clicked(self):
        try :
            import tensorflow as tf
        except ImportError:
            self.print("TensorFlow is not available. Correction cannot be performed.")
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("TensorFlow Not Available")
            msg_box.setText("TensorFlow is not available. Correction cannot be performed.")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()
            return
            
        if not os.path.exists(self.tomo_path_and_stage.area_path):
            self.print("Please draw tomo area first.")
        elif not os.path.exists(self.tomo_path_and_stage.deconv_tomo_path):
            self.print("Predict need deconv data.")
        elif not os.path.exists(self.tomo_path_and_stage.isonet_tomo_path):
            self.print("Predict need correction data.")
        else:
            
            from qtpy.QtWidgets import QProgressDialog
            from qtpy.QtCore import Qt
            self.progress_dialog = QProgressDialog("Processing...", 'Cancel', 0, 100, self.main_viewer)
            self.progress_dialog.setWindowTitle('Predicting')
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.setValue(0)
            self.progress_dialog.show()
            self.deconv_data = self.viewer.layers['deconv_tomo'].data
            self.corrected_data = self.viewer.layers['corrected_tomo'].data
            self.progress_dialog.setValue(20)
            self.label = predict_label(self.deconv_data, self.corrected_data)
            # self.label = self.viewer.layers['label'].data
            self.progress_dialog.setValue(40)
            self.area_path = self.tomo_path_and_stage.area_path
            self.processed_vesicles, self.shape = morph_process(self.label, self.area_path)
            self.progress_dialog.setValue(60)

            self.vesicle_info = vesicle_measure(self.corrected_data, self.processed_vesicles, self.shape, min_radius=8)
            with open(self.tomo_path_and_stage.new_json_file_path,"w") as out:
                json.dump(self.vesicle_info,out)
            self.ves_tomo = vesicle_rendering(self.vesicle_info, self.shape)
            with mrcfile.new(self.tomo_path_and_stage.new_label_file_path, overwrite=True) as mrc:
                mrc.set_data(self.ves_tomo)
            self.viewer.add_labels(self.ves_tomo, name='label')
            self.viewer.layers['label'].opacity = 0.5 
            self.viewer.layers.selection.active = self.viewer.layers['edit vesicles']
            self.progress_dialog.setValue(100)
            self.show_current_state()
        
    def register_draw_memb_mod(self):
        # 保存为临时点文件并转换为 .mod 文件
        def write_model(model_file, model_df):
            """ 将点文件转换为 .mod 文件 """
            model = np.asarray(model_df)

            # 提取model_file的文件夹路径
            model_dir = os.path.dirname(model_file)

            # 如果文件夹不存在，则创建文件夹
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            with tempfile.NamedTemporaryFile(suffix=".pt", dir=".") as temp_file:
                # 保存点文件
                point_file = temp_file.name
                np.savetxt(point_file, model, fmt=(['%d']*2 + ['%.2f']*3))

                # 使用 point2model 命令将点文件转换为 .mod 文件
                cmd = f"point2model -op {point_file} {model_file} >/dev/null"
                subprocess.run(cmd, shell=True, check=True)

        def validate_points(points):
            """ 验证点是否满足要求 """
            # 按照 z 轴分组
            z_groups = {}
            for point in points:
                z = point[0]
                if z not in z_groups:
                    z_groups[z] = []
                z_groups[z].append(point)

            # 检查是否有四个不同的 z 轴
            if len(z_groups) != 4:
                self.print("Error: Points must be distributed across exactly four distinct Z-axes.")
                return None

            # 检查其中一个 z 轴上只有一个点，其他三个 z 轴上点的数量大于一个
            single_point_z_count = 0
            multi_point_z_count = 0
            for group in z_groups.values():
                if len(group) == 1:
                    single_point_z_count += 1
                elif len(group) > 1:
                    multi_point_z_count += 1

            if single_point_z_count != 1 or multi_point_z_count != 3:
                self.print("Error: There must be exactly one Z-axis with a single point and three Z-axes with more than one point.")
                return None

            return z_groups

        # 获取点数据
        points = self.viewer.layers['edit vesicles'].data

        # 验证点数据是否符合要求
        z_groups = validate_points(points)

        # 如果验证不通过，直接退出并清空点数据
        if z_groups is None:
            self.viewer.layers['edit vesicles'].data = None
            return

        # 准备保存的点数据
        data = []
        # 创建 object1 并添加三个 contour
        object_id = 1
        contour_count = 0
        for z, group in z_groups.items():
            if len(group) > 1 and contour_count < 3:
                for point in group:
                    # object_id, contour_id, x, y, z (1-based for object and contour)
                    data.append([object_id, contour_count + 1, point[2], point[1], point[0]])
                contour_count += 1

        # 创建 object2 并添加单独的点
        object_id = 2
        for z, group in z_groups.items():
            if len(group) == 1:
                for point in group:
                    # object_id, contour_id, x, y, z
                    data.append([object_id, 1, point[2], point[1], point[0]])

        # 转换为DataFrame
        df = pd.DataFrame(data, columns=["object", "contour", "x", "y", "z"])
        write_model(self.tomo_path_and_stage.memb_prompt_path, df)
        self.print("Points validated and saved successfully.")

        # 最后清空点数据
        self.viewer.layers['edit vesicles'].data = None
    
    def register_seg_memb(self):
        # 显示进度对话框
        from qtpy.QtWidgets import QProgressDialog
        from qtpy.QtCore import Qt
        self.progress_dialog = QProgressDialog("Processing...", 'Cancel', 0, 100, self.main_viewer)
        self.progress_dialog.setWindowTitle('Opening')
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setValue(0)
        self.progress_dialog.show()

        # 设置输出路径
        output_path = self.tomo_path_and_stage.memb_folder_path + '/' + self.tomo_path_and_stage.base_tomo_name
        cmd = f'segprepost.py run {self.tomo_path_and_stage.isonet_tomo_path} {self.tomo_path_and_stage.memb_prompt_path} -o {output_path}'

        try:
            # 运行命令并捕获错误
            subprocess.run(cmd, shell=True, check=True)

            # 成功时的提示信息
            result = f'Membrane segmentation successful. The result is saved at {self.tomo_path_and_stage.memb_folder_path}. You can click Visualize to view the result.'
            self.print(result)

        except subprocess.CalledProcessError as e:
            # 捕获错误并输出错误信息
            error_message = f"An error occurred during membrane segmentation: {str(e)}"
            self.print(error_message)

        # 进度完成
        self.progress_dialog.setValue(100)
    
    def register_vis_memb(self):
        def read_point(point_file, dtype_z=int):
            """ Read imod point file. """
            point = np.loadtxt(point_file)
            if point.shape[1] != 5:
                raise ValueError("Point file should contain five columns, corresponding to object,contour,x,y,z.")
            
            cols = ["object", "contour", "x", "y", "z"]
            dtypes = [int, int, float, float, dtype_z]
            data = {
                cols[i]: pd.Series(point[:, i], dtype=dtypes[i])
                for i in range(5)
            }
            model = pd.DataFrame(data)
            return model

        def read_model(model_file, dtype_z=int):
            """ Read imod model file. """
            with tempfile.NamedTemporaryFile(suffix=".pt", dir=".") as temp_file:
                point_file = temp_file.name
                # cmd = f"model2point -ob {model_file} {point_file}"
                cmd = f"model2point -ob {model_file} {point_file} >/dev/null"
                subprocess.run(cmd, shell=True, check=True)
                
                bak_file = Path(f"{point_file}~")
                if bak_file.exists():
                    bak_file.unlink()

                model = read_point(point_file, dtype_z=dtype_z)
                return model

        def mod_to_3d_data(model_df, shape):
            """ Convert model DataFrame to 3D numpy array. """
            data = np.zeros(shape, dtype=np.int16)
            
            # Mapping from object values to desired output values
            obj_mapping = {1: 0, 2: 2, 3: 0}
            # obj_mapping = {1: 37, 2: 2, 3: 22}
            
            for _, row in model_df.iterrows():
                obj = int(row['object'])
                z, y, x = int(row['z']), int(row['y']), int(row['x'])
                if obj in obj_mapping:
                    data[z, y, x] = obj_mapping[obj]
                    
            return data

        model_file = self.tomo_path_and_stage.memb_result_path
        if not os.path.exists(model_file):
            result = f'Error: The file {model_file} does not exist.'
            self.print(result)
            return
        
        model_df = read_model(model_file)

        shape = self.viewer.layers[0].data.shape  # 请根据你的实际情况调整
        data = mod_to_3d_data(model_df, shape)

        # 使用Napari可视化
        self.viewer.add_labels(data, name='Membrane', opacity=1)

        # 操作成功后输出提示
        self.print("Visualize successfully.")
    
    def register_draw_area_mod(self):
        
        def validate_points(points):
            if len(points) == 0 :
                return False
            # 获取所有z值
            z_values = points[:, 2]
            unique_z_values, counts = np.unique(z_values, return_counts=True)

            # 检查z值的条件
            if len(unique_z_values) != 3:
                return False

            # 查找中间z值
            middle_z = unique_z_values[1]

            # 检查z值是否满足条件
            if counts[0] != 1 or counts[2] != 1:
                return False

            return True
        
        
        def create_area_mod():
            points = self.viewer.layers['edit vesicles'].data  # (z, y, x) 格式
            # 转换成 (x, y, z) 形式
            points_transformed = np.array([[x, y, z] for z, y, x in points])
            # 检查点的形式
            if validate_points(points_transformed):
                # 保存路径
                point_file_path = os.path.join(self.tomo_path_and_stage.root_dir, 'points.point')
                # output_mod_file = os.path.join(self.tomo_path_and_stage.root_dir, 'area.mod')
                output_mod_file = self.tomo_path_and_stage.area_path

                # 保存到 .point 文件，以更可读的形式
                with open(point_file_path, 'w') as file:
                    for point in points_transformed:
                        file.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")

                # 通过命令行执行 model2point 命令并保存 area.mod 文件
                subprocess.run(['point2model', point_file_path, output_mod_file, '-planar'])
                self.viewer.layers['edit vesicles'].data = None
                
                if os.path.exists(point_file_path):
                    os.remove(point_file_path)
                
                self.print("Points validated and saved successfully.")
            else:
                self.print("Points do not meet the required conditions.")
        
        try:
            self.toolbar_widget.draw_tomo_area_button.clicked.disconnect()
        except TypeError:
            pass
        self.toolbar_widget.draw_tomo_area_button.clicked.connect(create_area_mod)
        
        
        
    def register_manualy_correction(self):
        
        def init_lable_file():
            if self.viewer.layers:
                label_shape = self.viewer.layers[0].data.shape
                zero_label = np.zeros(label_shape, dtype=np.int16)
                with mrcfile.new(self.tomo_path_and_stage.new_label_file_path, overwrite=True) as mrc:
                    mrc.set_data(zero_label.astype(np.float32))
                
                self.viewer.add_labels(zero_label, name='label')  # add label layer
                self.viewer.layers['label'].opacity = 0.5
                
                self.viewer.layers.selection.active = self.viewer.layers['edit vesicles']
                # 将初始数据写入文件
                initial_data = {
                    "vesicles": []
                }
                with open(self.tomo_path_and_stage.new_json_file_path, 'w') as json_file:
                    json.dump(initial_data, json_file, indent=4)
                self.multiple_viewer_widget.utils_widget.ui.progressStage.setText('Please Make Manualy Annotations')
        
        try:
            self.toolbar_widget.manual_annotation_button.clicked.disconnect()
        except TypeError:
            pass
        self.toolbar_widget.manual_annotation_button.clicked.connect(init_lable_file)