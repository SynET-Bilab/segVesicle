import numpy as np
import mrcfile
import napari
import os
import subprocess
import json
import tempfile

import xml.etree.ElementTree as ET

from qtpy import QtCore, QtWidgets
from qtpy.QtWidgets import (
    QLabel, QLineEdit, QPushButton, 
    QVBoxLayout, QHBoxLayout, QMessageBox
)

from napari.utils.notifications import show_info
import pandas as pd
from pathlib import Path

from three_orthos_viewer import CrossWidget, MultipleViewerWidget
from tomo_path_and_stage import TomoPathAndStage
from qtpy.QtWidgets import QFileDialog, QDialog, QVBoxLayout, QPushButton, QLineEdit, QLabel, QHBoxLayout, QMessageBox

from util.distance_calculator import distance_calc
# from window.vesicle_window import VesicleViewer
from window.deconv_window import DeconvWindow
from window.correction_window import CorrectionWindow
from window.memb_segmentation_window import MembSegmentationWindow
from window.distance_filter_window import DistanceFilterWindow
from window.annotate_vesicle_class import VesicleAnnotationWindow
from window.fix_false_negative_vesicle import FixFNWindow
from window.xml_exporter_dialog import export_final_xml
from window.interpolate_memb_dialog import InterpolateMembDialog
from util.wd_excel_export import export_wd_excel
from window.finetune_model_window import FinetuneModelWindow
from util.add_layer_with_right_contrast import add_layer_with_right_contrast
from util.predict_vesicle import predict_label, morph_process, vesicle_measure, vesicle_rendering
from util.resample import resample_image
from util.json2xlsx import json_to_excel
from util.io import get_tomo
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
        self.register_export_xlsx()
        self.register_finetune_model()
        self.register_manualy_correction()
        
        self.register_distance_calc()
        self.register_filter_vesicle()
        self.register_annotate_pit()
        self.register_annotate_vesicle()
        self.register_multi_class_visualize()
        self.register_export_final_xml()
        
        self.register_manualy_draw_memb()
        # self.register_extract_memb()
        self.register_interpolate_memb()
        self.register_save_mamually_memb()
        
        # self.register_analyze_by_volume()
        # self.register_show_single_vesicle()
        self.register_fix_fn_vesicle()
        self.register_xport_wd_excel()
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
        self.toolbar_widget.stsyseg_button.clicked.connect(self.open_segmentation_window)
        
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
                
                # 打印成功信息并告知路径
                resampled_image_path = self.tomo_path_and_stage.rec_tomo_path
                message = f"Resampled image successfully saved at: {resampled_image_path}"
                self.print(message)
                
                # 执行创建 tomograms.star 文件
                try:
                    # 获取 rec_tomo_path 所在的文件夹路径
                    rec_tomo_dir = os.path.dirname(self.tomo_path_and_stage.rec_tomo_path)
                    
                    # 获取 tomograms_star_path
                    output_star = self.tomo_path_and_stage.tomograms_star_path
                    
                    # 构建命令
                    command = [
                        'isonet.py', 'prepare_star', rec_tomo_dir,
                        '--pixel_size', '17.14',
                        '--output_star', output_star
                    ]
                    
                    # 执行命令
                    subprocess.run(command, check=True)
                    
                    # 成功提示
                    self.print('Tomo star created successfully!')
                
                except subprocess.CalledProcessError as e:
                    # 错误提示
                    self.print('Failed to create tomo star.')
                
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
            
        if self.viewer.layers != None:
            
            from qtpy.QtWidgets import QProgressDialog
            from qtpy.QtCore import Qt
            self.progress_dialog = QProgressDialog("Processing...", 'Cancel', 0, 100, self.main_viewer)
            self.progress_dialog.setWindowTitle('Predicting')
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.setValue(0)
            self.progress_dialog.show()
            
            # 检查并获取 deconv_data
            if 'deconv_tomo' in self.viewer.layers:
                self.deconv_data = self.viewer.layers['deconv_tomo'].data
            elif os.path.exists(self.tomo_path_and_stage.deconv_tomo_path):
                self.deconv_data = get_tomo(self.tomo_path_and_stage.deconv_tomo_path)
            else:
                self.deconv_data = self.viewer.layers[0].data

            # 检查并获取 corrected_data
            if 'corrected_tomo' in self.viewer.layers:
                self.corrected_data = self.viewer.layers['corrected_tomo'].data
            elif os.path.exists(self.tomo_path_and_stage.isonet_tomo_path):
                self.corrected_data = get_tomo(self.tomo_path_and_stage.isonet_tomo_path)
            else:
                self.corrected_data = self.viewer.layers[0].data
            
            self.progress_dialog.setValue(20)
            self.label = predict_label(self.deconv_data, self.corrected_data)
            # self.label = self.viewer.layers['label'].data
            self.progress_dialog.setValue(40)
            if os.path.exists(self.tomo_path_and_stage.area_by_sam_path):
                self.area_path = self.tomo_path_and_stage.area_by_sam_path
            elif os.path.exists(self.tomo_path_and_stage.area_path):
                self.area_path = self.tomo_path_and_stage.area_path
            else:
                self.area_path = None
            
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
        
        
    def register_export_xlsx(self):
        def export_xlsx():
            try:
                # Attempt to execute the JSON to Excel conversion
                json_to_excel(self.tomo_path_and_stage.new_json_file_path, 
                            self.tomo_path_and_stage.xlsx_file_path)
                # Inform the user of success
                self.print(f"Excel file successfully saved at {self.tomo_path_and_stage.xlsx_file_path}.")
            except Exception as e:
                # Inform the user of any error that occurs
                self.print(f"Failed to export Excel file: {str(e)}")
        try:
            self.toolbar_widget.export_xlsx_button.clicked.disconnect()
        except TypeError:
            pass
        
        self.toolbar_widget.export_xlsx_button.clicked.connect(export_xlsx)
    
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
    
    def open_segmentation_window(self):
        # 创建并显示用于输入 pixel_size 和 extend 的新窗口
        self.segmentation_window = MembSegmentationWindow(self)
        self.segmentation_window.show()
    
    
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
            obj_mapping = {1: 0, 2: 2, 3: 22}
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
        self.memb_df = model_df

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
        
    def register_manualy_draw_memb(self):
        def manualy_draw_memb():
            # Step 1: Get the shape of the first layer
            label_shape = self.viewer.layers[0].data.shape
            
            # Step 2: Create an array of zeros with the same shape
            zero_array = np.zeros(label_shape, dtype=np.uint8)
            
            # Step 3: Add the array as a new layer with the name 'Draw Membrane'
            membrane_layer = self.viewer.add_labels(zero_array, name='Draw Membrane')
            
            # Step 4: Set the layer to paint mode with brush size of 1
            membrane_layer.mode = 'paint'
            membrane_layer.brush_size = 1
    
        try:
            self.toolbar_widget.manualy_draw_button.clicked.disconnect()
        except TypeError:
            pass
        self.toolbar_widget.manualy_draw_button.clicked.connect(manualy_draw_memb)
        
    def register_extract_memb(self):
        def extract_memb():
            """
            提取并处理膜层（Membrane）的mask数据。

            参数：
            - self: 类的实例，包含viewer和layers属性。
            - extract_stride (int): 切片步长，每隔多少个z切片保留一次mask。
            """
            extract_stride = 10
            
            # 获取原始的三维mask数据 (z, y, x)
            original_data = self.viewer.layers['Membrane'].data

            # 步骤 2: 只保留mask值为2的部分，其余设置为0
            # 生成一个与original_data形状相同的布尔数组，值为True的位置对应mask值为2
            mask_value_2 = (original_data == 2)
            
            # 创建一个新的数据数组，初始为全0
            new_data = np.zeros_like(original_data, dtype=original_data.dtype)
            
            # 将mask值为2的位置设置为2，其他位置保持为0
            new_data[mask_value_2] = 1

            # 步骤 3: 设置切片步长extract_stride
            # 找到第一个z切片中mask不是全0的位置
            # 通过对每个z切片求和，判断是否有非零值
            z_sums = new_data.sum(axis=(1, 2))
            non_zero_z_indices = np.where(z_sums > 0)[0]
            
            if non_zero_z_indices.size == 0:
                print("没有找到任何mask值为2的z切片。")
                # 如果没有找到非零的z切片，直接覆盖原数据
                self.viewer.layers['Membrane'].data = new_data
                return
            
            # 第一个非零的z切片索引
            start_z = non_zero_z_indices[0]
            
            # 生成一个布尔数组，标记需要保留的z切片
            # 从start_z开始，每隔extract_stride个切片保留一次
            mask_z = np.zeros(new_data.shape[0], dtype=bool)
            mask_z[start_z::extract_stride] = True
            
            # 应用z切片的mask，只保留标记为True的切片，其他切片置0
            # 使用广播机制，将mask_z应用到所有y和x维度
            new_data = np.where(mask_z[:, None, None], new_data, 0)
            
            # 步骤 4: 覆盖原始的mask数据
            self.viewer.layers['Membrane'].data = new_data

            print("膜层mask数据已成功提取并更新。")
    
        try:
            self.toolbar_widget.extract_memb_button.clicked.disconnect()
        except TypeError:
            pass
        self.toolbar_widget.extract_memb_button.clicked.connect(extract_memb)
        
    def register_interpolate_memb(self):
        def on_interpolate_memb_button_clicked():
            dialog = InterpolateMembDialog(parent=self.viewer.window._qt_window)
            if dialog.exec_():
                operation, membrane_type, threshold = dialog.get_values()
                try:
                    interpolate_memb(operation, membrane_type, threshold)
                except Exception as e:
                    QMessageBox.critical(self.viewer.window._qt_window, "Error", str(e))
                    
        def interpolate_memb(operation='dilation_erosion', membrane_type='front', threshold=0.2):
            from scipy.ndimage import binary_dilation, binary_erosion
            """
            Interpolates membrane lines across multiple z-layers and creates a new label layer in Napari.
            """
            # Retrieve existing membrane mask data, assuming shape is (z, y, x)
            if 'Draw Membrane' not in self.viewer.layers:
                raise ValueError("Layer 'Draw Membrane' not found.")

            membrane_mask = self.viewer.layers['Draw Membrane'].data

            if membrane_mask.ndim != 3:
                raise ValueError("Data in 'Draw Membrane' layer should be a 3D array (z, y, x).")

            # Determine which z-layers contain membrane annotations
            labeled_z = np.where(membrane_mask.any(axis=(1, 2)))[0]
            labeled_z = np.sort(labeled_z)

            if len(labeled_z) < 2:
                raise ValueError("At least two z-layers with membrane annotations are required for interpolation.")

            # Initialize interpolated membrane mask with all zeros
            interpolated_mask = np.zeros_like(membrane_mask, dtype=np.uint8)

            structuring_element = np.ones((3, 3), dtype=bool)  # 3x3 structuring element

            # Perform morphological operations based on selected operation type
            if operation == 'dilation_erosion':
                # Apply dilation before interpolation
                for z in labeled_z:
                    membrane_mask[z] = binary_dilation(membrane_mask[z], structure=structuring_element)
            elif operation == 'none':
                pass  # No morphological operations
            else:
                raise ValueError("Unknown operation type")

            # Copy existing membrane annotations to the interpolated mask
            interpolated_mask[labeled_z] = membrane_mask[labeled_z]

            # Iterate through adjacent labeled z-layers and perform linear interpolation
            for i in range(len(labeled_z) - 1):
                z_start = labeled_z[i]
                z_end = labeled_z[i + 1]
                mask_start = membrane_mask[z_start].astype(float)
                mask_end = membrane_mask[z_end].astype(float)
                num_interp = z_end - z_start - 1

                if num_interp <= 0:
                    continue  # Adjacent z-layers, no interpolation needed

                for dz in range(1, num_interp + 1):
                    z_interp = z_start + dz
                    alpha = dz / (z_end - z_start)  # Interpolation weight

                    # Perform linear interpolation
                    mask_interp = (1 - alpha) * mask_start + alpha * mask_end

                    # Binarize the interpolation result
                    mask_interp_binary = (mask_interp > threshold).astype(np.uint8)

                    # Update the interpolated mask with binary values
                    interpolated_mask[z_interp] = mask_interp_binary

            # Perform morphological operations after interpolation
            if operation == 'dilation_erosion':
                # Apply erosion after interpolation
                for z in range(interpolated_mask.shape[0]):
                    if interpolated_mask[z].any():  # If the current layer has annotations
                        interpolated_mask[z] = binary_erosion(interpolated_mask[z], structure=structuring_element) * interpolated_mask[z]
            elif operation == 'none':
                pass  # No morphological operations
            else:
                raise ValueError("Unknown operation type")

            # Update the interpolated mask with appropriate values based on membrane type
            if membrane_type == 'front':
                interpolated_mask[interpolated_mask > 0] = 2
            elif membrane_type == 'rear':
                interpolated_mask[interpolated_mask > 0] = 22
            else:
                raise ValueError("Unknown membrane type")

            # Add the new label layer to Napari
            if 'Membrane Labels' in self.viewer.layers:
                self.viewer.layers.remove('Membrane Labels')  # Remove old layer if it exists

            self.viewer.add_labels(interpolated_mask, name='Membrane Labels')
            self.viewer.layers.remove('Draw Membrane')
                
        try:
            self.toolbar_widget.interpolate_memb_button.clicked.disconnect()
        except TypeError:
            pass
        self.toolbar_widget.interpolate_memb_button.clicked.connect(on_interpolate_memb_button_clicked)
    
    def register_save_mamually_memb(self):
        def save_mamually_memb():
            original_data = self.viewer.layers['Membrane'].data
            new_data = self.viewer.layers['Membrane Labels'].data
            # 进行按位或运算
            combined_data = original_data | new_data

            # 将结果覆盖回原来的层
            self.viewer.layers['Membrane'].data = combined_data

            # 可选：更新显示
            self.viewer.layers['Membrane'].refresh()
            
            self.viewer.layers.remove('Membrane Labels')
            
            # 1. 获取修改后的数据
            data = self.viewer.layers['Membrane'].data
            
            # 2. 定义反向映射关系，从数据值映射回object编号
            # 原始的obj_mapping是 {1: 0, 2: 2, 3: 22}
            # 因此反向映射为 {2: 2, 22: 3}
            obj_mapping_reverse = {2: 2, 22: 3}
            
            # 3. 提取非零值的坐标和对应的object编号
            z_idx, y_idx, x_idx = np.nonzero(data)
            values = data[z_idx, y_idx, x_idx]
            
            # 仅保留在反向映射中的值（即排除值为0或不在映射中的值）
            valid_indices = [i for i, val in enumerate(values) if val in obj_mapping_reverse]
            z_idx = z_idx[valid_indices]
            y_idx = y_idx[valid_indices]
            x_idx = x_idx[valid_indices]
            values = values[valid_indices]
            
            # 获取对应的object编号
            objects = [obj_mapping_reverse[val] for val in values]
            
            # 4. 调整contour和z的值
            # contour代表点所在的不同z轴，从1开始
            contours = z_idx + 1  # 将z_idx从0开始调整为从1开始
            z_values = z_idx.astype(int)  # z值为索引即可，无需调整
            
            # 5. 构建DataFrame
            df = pd.DataFrame({
                'object': objects,
                'contour': contours,
                'x': x_idx.astype(float),
                'y': y_idx.astype(float),
                'z': z_values
            })
            
            # 6. 保存DataFrame为点文件，并转换回.mod文件
            with tempfile.NamedTemporaryFile(suffix=".pt", dir=".") as temp_file:
                point_file = temp_file.name
                # 保存点文件
                df.to_csv(point_file, sep=' ', index=False, header=False)
                
                # 使用point2model将点文件转换为.mod文件
                output_mod_file = self.tomo_path_and_stage.new_memb_result_path
                cmd = f"point2model {point_file} {output_mod_file} >/dev/null"
                subprocess.run(cmd, shell=True, check=True)
            
            # 7. 操作成功后输出提示
            self.print(f"Membrane layer saved back to {output_mod_file} successfully.")
            
        
        try:
            self.toolbar_widget.save_memb_button.clicked.disconnect()
        except TypeError:
            pass
        self.toolbar_widget.save_memb_button.clicked.connect(save_mamually_memb)
        
    def register_finetune_model(self):
        def open_finetune_window():
            finetune_window = FinetuneModelWindow(parent=self.main_viewer, tomo_path_and_stage=self.tomo_path_and_stage, print_callback=self.print)
            finetune_window.exec_()
        
        try:
            self.toolbar_widget.finetune_model_button.clicked.disconnect()
        except TypeError:
            pass
        self.toolbar_widget.finetune_model_button.clicked.connect(open_finetune_window)
        
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
        
    def register_distance_calc(self):
        
        def distance_calculation():
            # Define the file paths
            json_path = self.tomo_path_and_stage.new_json_file_path
            mod_path = self.tomo_path_and_stage.memb_result_path
            xml_output_path = self.tomo_path_and_stage.ori_xml_path

            # Call the distance_calc function, passing the paths and a print function
            distance_calc(json_path, mod_path, xml_output_path, self.print)

        
        try:
            self.toolbar_widget.distance_calc_button.clicked.disconnect()
        except TypeError:
            pass
        self.toolbar_widget.distance_calc_button.clicked.connect(distance_calculation)
        
    
    def register_filter_vesicle(self):
        
        def filter_vesicle():
            dialog = DistanceFilterWindow(self)
            dialog.exec_()

        
        try:
            self.toolbar_widget.filter_by_distance_button.clicked.disconnect()
        except TypeError:
            pass
        self.toolbar_widget.filter_by_distance_button.clicked.connect(filter_vesicle)
        
    def register_annotate_vesicle(self):
        
        def annotate_vesicle():
            window = VesicleAnnotationWindow(self)
            window.show()

        try:
            self.toolbar_widget.annotate_vesicle_type_button.clicked.disconnect()
        except TypeError:
            pass
        self.toolbar_widget.annotate_vesicle_type_button.clicked.connect(annotate_vesicle)
        
    def register_fix_fn_vesicle(self):
        
        def fix_fn_vesicle():
            
            def verify_files_exist():
                # Check if the required XML paths exist
                class_xml_path = self.tomo_path_and_stage.class_xml_path
                ori_filter_xml_path = self.tomo_path_and_stage.ori_filter_xml_path
                filter_xml_path = self.tomo_path_and_stage.filter_xml_path
                return all(os.path.exists(path) for path in [class_xml_path, ori_filter_xml_path, filter_xml_path])
                # Perform checks before creating FixFNWindow
                
            def filter_vesicle_ids():
                # Load ori_filter_xml and filter_xml files
                ori_tree = ET.parse(self.tomo_path_and_stage.ori_filter_xml_path)
                ori_root = ori_tree.getroot()
                filter_tree = ET.parse(self.tomo_path_and_stage.filter_xml_path)
                filter_root = filter_tree.getroot()

                # Find vesicle IDs that are Type='vesicle' in ori_filter_xml and Type='other' in filter_xml
                ori_vesicles = {
                    vesicle.attrib['vesicleId']: vesicle.find('Type').attrib.get('t')
                    for vesicle in ori_root.findall('Vesicle') 
                    if vesicle.find('Type') is not None
                }
                filter_vesicles = {
                    vesicle.attrib['vesicleId']: vesicle.find('Type').attrib.get('t')
                    for vesicle in filter_root.findall('Vesicle') 
                    if vesicle.find('Type') is not None
                }

                # Identify IDs to be annotated
                vesicle_ids = [vesicle_id for vesicle_id, v_type in ori_vesicles.items()
                            if v_type == 'vesicle' and filter_vesicles.get(vesicle_id) == 'others']
                return vesicle_ids
                
            if not verify_files_exist():
                QMessageBox.information(None, "Notice", "No need to modify categories.")
                return
            
            vesicle_ids = filter_vesicle_ids()
            if not vesicle_ids:
                QMessageBox.information(None, "Notice", "No need to modify categories.")
                return
            window = FixFNWindow(self)
            window.show()

        try:
            self.toolbar_widget.fix_fn_button.clicked.disconnect()
        except TypeError:
            pass
        self.toolbar_widget.fix_fn_button.clicked.connect(fix_fn_vesicle)
        
    def register_multi_class_visualize(self):
        def multi_class_visualize():
            print("Starting multi-class visualization...")
            try:
                # 解析 XML 文件
                tree = ET.parse(self.tomo_path_and_stage.class_xml_path)
                root = tree.getroot()

                # 定义类型映射
                type_mapping = {
                    'false': 0,
                    'tether': 1,
                    'contact': 2,
                    'omega': 3,
                    'pit': 4,
                    'CCV': 5,
                    'endosome': 6,
                    'DCV': 7,
                    'others': 8
                }

                # 定义要包含的类型
                included_types = list(type_mapping.keys())

                # 初始化新的标签数组
                label_data = self.viewer.layers['label'].data
                new_label = np.zeros_like(label_data, dtype=np.int32)

                # 遍历 XML 中的每个囊泡
                for vesicle in root.findall('Vesicle'):
                    vesicle_type_element = vesicle.find('Type')
                    if vesicle_type_element is not None:
                        vesicle_type = vesicle_type_element.get('t')
                        if vesicle_type in included_types:
                            # 获取中心坐标
                            center = vesicle.find('Center')
                            if center is not None:
                                try:
                                    x = float(center.attrib['X'])
                                    y = float(center.attrib['Y'])
                                    z = float(center.attrib['Z'])
                                except (ValueError, KeyError) as coord_error:
                                    print(f"Invalid center coordinates for vesicle: {coord_error}")
                                    continue

                                # 将物理坐标转换为体素索引（假设坐标已经是索引，若需要转换请根据实际情况调整）
                                z_idx = int(round(z))
                                y_idx = int(round(y))
                                x_idx = int(round(x))

                                # 检查索引是否在 label_data 的范围内
                                if (0 <= z_idx < label_data.shape[0] and
                                    0 <= y_idx < label_data.shape[1] and
                                    0 <= x_idx < label_data.shape[2]):
                                    
                                    vesicle_label = label_data[z_idx, y_idx, x_idx]
                                    
                                    if vesicle_label != 0:
                                        mapped_value = type_mapping[vesicle_type]
                                        # 将所有对应 vesicle_label 的像素设置为 mapped_value
                                        new_label[label_data == vesicle_label] = mapped_value
                                        print(f"Vesicle at ({z_idx}, {y_idx}, {x_idx}) with label {vesicle_label} of type '{vesicle_type}' mapped to {mapped_value}.")
                                    else:
                                        print(f"No label found at center coordinates ({z_idx}, {y_idx}, {x_idx}) for vesicle type '{vesicle_type}'.")
                                else:
                                    print(f"Center coordinates ({z}, {y}, {x}) out of bounds.")
                            else:
                                print("No Center element found for a vesicle.")
                
                # 添加新的多分类标签层
                self.viewer.add_labels(new_label, name='multi_class_labels')

                # 隐藏原始标签层
                self.viewer.layers['label'].visible = False

                self.print("Multi-class visualization completed successfully.")

            except ET.ParseError as parse_err:
                print(f"XML parsing error: {parse_err}")
            except FileNotFoundError as fnf_error:
                print(f"XML file not found: {fnf_error}")
            except Exception as e:
                print(f"An error occurred during multi-class visualization: {e}")

        try:
            self.toolbar_widget.multi_class_visualize_button.clicked.disconnect()
        except TypeError:
            pass
        self.toolbar_widget.multi_class_visualize_button.clicked.connect(multi_class_visualize)


    def register_annotate_pit(self):
        def annotate_pit():
        # Step 1: Get the three point coordinates ABC(z, y, x) from 'edit vesicles' layer
            if 'edit vesicles' not in self.viewer.layers:
                self.print("Error: 'edit vesicles' layer not found.")
                return

            points = self.viewer.layers['edit vesicles'].data
            if len(points) < 3:
                self.print("Error: Need at least three points to define a pit.")
                self.viewer.layers['edit vesicles'].data = None
                return

            # Get the last three points as A, B, and C
            A = np.array(points[-3])
            B = np.array(points[-2])
            C = np.array(points[-1])
            self.viewer.layers['edit vesicles'].data = None
            
            # Step 2: Read VesicleList from XML
            from util.structures import Vesicle, VesicleList
            import os

            vl = VesicleList()
            xml_path = None

            if os.path.exists(self.tomo_path_and_stage.class_xml_path):
                xml_path = self.tomo_path_and_stage.class_xml_path
            elif os.path.exists(self.tomo_path_and_stage.filter_xml_path):
                xml_path = self.tomo_path_and_stage.filter_xml_path
            else:
                self.print("Error: No XML file found.")
                return

            vl.fromXMLFile(xml_path)

            # Step 3: Calculate the vesicle information
            sv = Vesicle()
            # Assume A and C are the base, B is the tip
            sv.setType('pit')
            center = (A + C) / 2  # Compute the center point
            sv.setCenter(center[::-1])  # Reverse to (x, y, z) if necessary

            # Set B as PitPoint (reverse from z, y, x to x, y, z)
            pit_point = B[::-1]  # B is currently (z, y, x), we reverse it to (x, y, z)
            sv.setPitPoint(pit_point)  # Store B as PitPoint

            # Calculate the radius
            radius = np.linalg.norm(A - center)
            sv.setRadius(radius)
            sv.setRadius2D(np.array([radius, radius]))

            sv.setDistance(0.0)
            sv.setProjectionPoint(sv.getCenter())

            base_vector = C[[1, 2]] - A[[1, 2]]  # 只提取 y 和 x 分量
            rotation_2d = np.arctan2(base_vector[0], base_vector[1])  # y分量在前，x分量在后，弧度是相对于 x 轴
            # rotation_2d_deg = np.degrees(rotation_2d)  # 将弧度制转换为角度制
            sv.setRotation2D(rotation_2d)

            # Set the vesicle ID
            new_id = max(vl[-1].getId() + 1, 10000)
            sv.setId(new_id)

            vl.append(sv)

            # 保存 ABC 三点坐标到 .point 文件
            try:
                # 获取 XML 文件所在目录
                xml_dir = os.path.dirname(xml_path)

                # 定义 pits 子文件夹路径
                pits_dir = os.path.join(xml_dir, 'pits')

                # 如果 pits 文件夹不存在，则创建
                os.makedirs(pits_dir, exist_ok=True)

                # 定义文件名和完整路径
                filename = f"pit_1714_{new_id}.point"
                file_path = os.path.join(pits_dir, filename)

                # 将 A, B, C 点反转为 (x, y, z) 并保存为无标签、无括号格式
                with open(file_path, 'w') as f:
                    for point in [A, B, C]:
                        x, y, z = point[::-1]
                        f.write(f"{x} {y} {z}\n")

                self.print(f"Pit points saved to {file_path}")
            except Exception as e:
                self.print(f"Error saving pit points: {e}")

            # Step 4: Save the new XML to class_xml_path
            vl.toXMLFile(self.tomo_path_and_stage.class_xml_path)

            # Step 5: Add a new label layer to display the pit
            # Get the shape of the reference image layer (assumed to be the first layer)
            image_shape = self.viewer.layers[0].data.shape
            pit_layer_data = np.zeros(image_shape, dtype=np.uint8)

            # Check if the three points have the same z-coordinate
            if A[0] == B[0] and B[0] == C[0]:
                z_slice = int(A[0])

                # Enclose the points in the xy-plane and set the enclosed area to 1
                from skimage.draw import polygon
                rr, cc = polygon([A[1], B[1], C[1]], [A[2], B[2], C[2]], shape=pit_layer_data.shape[1:])
                pit_layer_data[z_slice, rr, cc] = 1
            else:
                # Points have different z, create a 3D cylinder from min to max z
                min_z = int(min(A[0], B[0], C[0]))
                max_z = int(max(A[0], B[0], C[0])) + 1  # Include max_z

                # Enclose the points in the xy-plane
                from skimage.draw import polygon
                rr, cc = polygon([A[1], B[1], C[1]], [A[2], B[2], C[2]], shape=pit_layer_data.shape[1:])

                # Set the enclosed area to 1 across the z-range
                pit_layer_data[min_z:max_z, rr, cc] = 1

            # 检查是否已经存在名为 'pit' 的标签层
            if 'pit' in self.viewer.layers:
                # 获取现有的 'pit' 层
                pit_layer = self.viewer.layers['pit'].data
                
                # 将新的 pit_layer_data 叠加到现有的层数据上
                updated_data = np.maximum(pit_layer, pit_layer_data)
                
                # 更新 'pit' 层的数据
                self.viewer.layers['pit'].data = updated_data
            else:
                # 如果 'pit' 层不存在，添加新的标签层
                self.viewer.add_labels(pit_layer_data, name='pit')

            if 'edit vesicles' in self.viewer.layers:
                self.viewer.layers.selection.active = self.viewer.layers['edit vesicles']
            
            # Step 6: Display a success message
            self.print("Pit added successfully.")

        # Step 7: Unbind 'p' and bind it to annotate_pit
        self.viewer.bind_key('p', None)  # Unbind existing 'p' key
        self.viewer.bind_key('p', lambda viewer: annotate_pit())  # Bind 'p' to annotate_pit

        try:
            self.toolbar_widget.annotate_pit_button.clicked.disconnect()
        except TypeError:
            pass
        self.toolbar_widget.annotate_pit_button.clicked.connect(annotate_pit)

    def register_export_final_xml(self):
        try:
            self.toolbar_widget.export_final_xml_button.clicked.disconnect()
        except TypeError:
            pass
        self.toolbar_widget.export_final_xml_button.clicked.connect(
            lambda: export_final_xml(self.main_viewer, self.tomo_path_and_stage, self.print)
        )
        
    def register_xport_wd_excel(self):
        try:
            self.toolbar_widget.export_wd_excel_button.clicked.disconnect()
        except TypeError:
            pass
        self.toolbar_widget.export_wd_excel_button.clicked.connect(
            lambda: export_wd_excel(self.tomo_path_and_stage, self.print)
        )