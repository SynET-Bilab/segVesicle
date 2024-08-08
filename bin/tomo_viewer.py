import numpy as np
import mrcfile
import napari
import os
import subprocess
import json

from qtpy import QtCore, QtWidgets
from napari.utils.notifications import show_info
import SimpleITK as sitk

from three_orthos_viewer import CrossWidget, MultipleViewerWidget
from tomo_path_and_stage import TomoPathAndStage
from qtpy.QtWidgets import QFileDialog, QInputDialog, QDialog, QVBoxLayout, QPushButton, QLineEdit, QLabel, QHBoxLayout


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
        progress_value = self.tomo_path_and_stage.progress_stage.value
        # 将进度状态展示在QLineEdit里
        self.multiple_viewer_widget.utils_widget.ui.progressStage.setText(progress_value)
    
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
                data = sitk.GetArrayFromImage(data)
                add_layer_with_right_contrast(data, 'ori_tomo', self.viewer)

                if 'corrected_tomo' in self.viewer.layers:
                    self.viewer.layers['corrected_tomo'].visible = False
                if 'edit vesicles' in self.viewer.layers:
                    self.viewer.layers.selection.active = self.viewer.layers['edit vesicles']
                
                self.viewer.layers.move(self.viewer.layers.index(self.viewer.layers['ori_tomo']), 0)
                self.viewer.add_points(name='edit vesicles', ndim=3, size=4)
                self.viewer.layers['edit vesicles'].mode = 'ADD'
                self.multiple_viewer_widget.viewer_model1.camera.zoom = 0.9
                self.multiple_viewer_widget.viewer_model2.camera.zoom = 0.9
                self.progress_dialog.setValue(100)
                message = f"Successfully opened the original image {file_path}."
                self.print(message)
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
        
        try:
            self.toolbar_widget.manual_annotation_button.clicked.disconnect()
        except TypeError:
            pass
        self.toolbar_widget.manual_annotation_button.clicked.connect(init_lable_file)
    