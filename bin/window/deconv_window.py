import napari
import numpy as np
import mrcfile
import os
import json
import subprocess
from qtpy.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit, QGridLayout, QDoubleSpinBox, QProgressDialog, QApplication
from qtpy.QtCore import Qt
from qtpy.QtGui import QScreen

from util.add_layer_with_right_contrast import add_layer_with_right_contrast
from util.deconvolution import deconv_tomo

class DeconvWindow(QMainWindow):
    def __init__(self, tomo_viewer):
        super().__init__(tomo_viewer.viewer.window.qt_viewer)
        self.setWindowTitle('Deconv Preview')
        
        self.tomo_viewer = tomo_viewer
        self.viewer = tomo_viewer.viewer
        self.preview_viewers = [napari.Viewer(show=False) for _ in range(11)]

        self.points = self.viewer.layers['edit vesicles'].data
        self.data = self.viewer.layers['ori_tomo'].data
        self.crop_data = self.get_cropped_data()
        
        # Default values
        self.default_values = {
            'voltage': 300.0,
            'cs': 2.7,
            'defocus': 0.0,
            'pixel_size': 17.14,
            'snrfalloff': 1.0,
            'deconvstrength': 1.0,
            'highpassnyquist': 0.02
        }

        # Deconv parameter inputs
        self.voltage_input = QDoubleSpinBox(self)
        self.voltage_input.setRange(100.0, 1000.0)
        self.voltage_input.setValue(self.default_values['voltage'])
        self.voltage_input.setSingleStep(10.0)  # 设置步长

        self.cs_input = QDoubleSpinBox(self)
        self.cs_input.setRange(0.0, 5.0)
        self.cs_input.setValue(self.default_values['cs'])
        self.cs_input.setSingleStep(0.1)  # 设置步长

        self.defocus_input = QDoubleSpinBox(self)
        self.defocus_input.setRange(0.0, 5.0)
        self.defocus_input.setValue(self.default_values['defocus'])
        self.defocus_input.setSingleStep(0.1)  # 设置步长

        self.pixel_size_input = QDoubleSpinBox(self)
        self.pixel_size_input.setRange(0.1, 100.0)
        self.pixel_size_input.setValue(self.default_values['pixel_size'])
        self.pixel_size_input.setSingleStep(1.0)  # 设置步长

        self.snrfalloff_input = QDoubleSpinBox(self)
        self.snrfalloff_input.setRange(0.0, 10.0)
        self.snrfalloff_input.setValue(self.default_values['snrfalloff'])
        self.snrfalloff_input.setSingleStep(0.1)  # 设置步长

        self.deconvstrength_input = QDoubleSpinBox(self)
        self.deconvstrength_input.setRange(0.0, 10.0)
        self.deconvstrength_input.setValue(self.default_values['deconvstrength'])
        self.deconvstrength_input.setSingleStep(0.1)  # 设置步长

        self.highpassnyquist_input = QDoubleSpinBox(self)
        self.highpassnyquist_input.setRange(0.0, 1.0)
        self.highpassnyquist_input.setValue(self.default_values['highpassnyquist'])
        self.highpassnyquist_input.setSingleStep(0.01)  # 设置步长

        self.save_button = QPushButton('Save Parameters', self)
        self.save_button.clicked.connect(self.save_parameters)

        self.apply_button = QPushButton('Apply Deconv', self)
        self.apply_button.clicked.connect(self.apply_deconv)
        
        self.preview_button = QPushButton('Preview Deconv', self)
        self.preview_button.clicked.connect(self.preview_deconv)

        layout = QGridLayout()

        # Add 3x4 grid of preview viewers
        snrfalloff_values = [0.3 + 0.1 * i for i in range(11)]
        for i, (viewer, snrfalloff) in enumerate(zip(self.preview_viewers, snrfalloff_values)):
            row = i // 4
            col = i % 4

            # 创建一个包含 VispyCanvas 的 QWidget 包装器
            container = QWidget()
            container_layout = QVBoxLayout()
            container_layout.addWidget(viewer.window.qt_viewer)
            container_layout.addWidget(QLabel(f'SNR Falloff: {snrfalloff:.2f}'))  # 将 QLabel 添加到 container_layout
            container.setLayout(container_layout)

            # 设置固定高度和宽度
            container.setFixedHeight(400)
            container.setFixedWidth(400)
            
            layout.addWidget(container, row, col)
        
        # Right bottom layout for parameter inputs
        right_bottom_layout = QVBoxLayout()
        self.add_parameter(right_bottom_layout, 'Voltage (kV):', self.voltage_input)
        self.add_parameter(right_bottom_layout, 'Cs (mm):', self.cs_input)
        self.add_parameter(right_bottom_layout, 'Defocus (um):', self.defocus_input)
        self.add_parameter(right_bottom_layout, 'Pixel Size (A):', self.pixel_size_input)
        self.add_parameter(right_bottom_layout, 'SNR Falloff:', self.snrfalloff_input)
        self.add_parameter(right_bottom_layout, 'Deconv Strength:', self.deconvstrength_input)
        self.add_parameter(right_bottom_layout, 'Highpass Nyquist:', self.highpassnyquist_input)
        right_bottom_layout.addWidget(self.save_button)
        right_bottom_layout.addWidget(self.preview_button)
        right_bottom_layout.addWidget(self.apply_button)
        
        # 将功能块添加到右下角
        layout.addLayout(right_bottom_layout, 2, 3, 2, 1)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.preview_deconv()
    
    def add_parameter(self, layout, label_text, widget):
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel(label_text))
        h_layout.addWidget(widget)
        layout.addLayout(h_layout)
    
    def get_cropped_data(self):
        p1 = self.points[0]  # 点的格式是 (z, y, x)
        p2 = self.points[1]
        z_val = int(round(p1[0]))  # 两个点的 Z 值相同
        min_y, min_x = np.min([p1[1:], p2[1:]], axis=0)
        max_y, max_x = np.max([p1[1:], p2[1:]], axis=0)
        min_y, min_x = int(min_y), int(min_x)
        max_y, max_x = int(max_y), int(max_x)
        min_z = max(0, z_val - 25)
        max_z = min(self.data.shape[0], z_val + 25 + 1)
        cropped_image = self.data[min_z:max_z, min_y:max_y + 1, min_x:max_x + 1]
        return cropped_image
    
    def save_parameters(self):
        # 获取当前参数值
        parameters = {
            'voltage': self.voltage_input.value(),
            'cs': self.cs_input.value(),
            'defocus': self.defocus_input.value(),
            'pixel_size': self.pixel_size_input.value(),
            'snrfalloff': round(self.snrfalloff_input.value(), 2), 
            'deconvstrength': self.deconvstrength_input.value(),
            'highpassnyquist': self.highpassnyquist_input.value()
        }

        # 保存为 JSON 文件
        save_path = self.tomo_viewer.tomo_path_and_stage.deconv_para  # 目标保存路径
        if not save_path.endswith('.json'):
            save_path += '.json'

        # 获取目录路径并判断是否存在，不存在则创建
        folder = os.path.dirname(save_path)
        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(save_path, 'w') as json_file:
            json.dump(parameters, json_file, indent=4)
        
        message = f"Parameters saved to {save_path}"
        print(message)
        self.tomo_viewer.print(message)
    
    def preview_deconv(self):
        self.progress_dialog = QProgressDialog("Processing...", 'Cancel', 0, 100, self)
        self.progress_dialog.setWindowTitle('Calculating')
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setValue(0)
        self.progress_dialog.show()
        
        # Move the progress dialog to the center of the screen
        screen_geometry = QScreen.availableGeometry(QApplication.primaryScreen())
        dialog_geometry = self.progress_dialog.geometry()
        x = (screen_geometry.width() - dialog_geometry.width()) // 2
        y = (screen_geometry.height() - dialog_geometry.height()) // 2
        self.progress_dialog.move(x, y)
        
        voltage = self.voltage_input.value()
        cs = self.cs_input.value()
        defocus = self.defocus_input.value()
        pixel_size = self.pixel_size_input.value()
        deconvstrength = self.deconvstrength_input.value()
        highpassnyquist = self.highpassnyquist_input.value()
        self.progress_dialog.setValue(50)
        snrfalloff_values = [0.3 + 0.1 * i for i in range(11)]
        for viewer, snrfalloff in zip(self.preview_viewers, snrfalloff_values):
            deconv_result = deconv_tomo(self.crop_data, None, angpix=pixel_size, voltage=voltage, cs=cs, defocus=defocus, 
                                        snrfalloff=snrfalloff, deconvstrength=deconvstrength, 
                                        highpassnyquist=highpassnyquist, phaseflipped=False, phaseshift=0, ncpu=4)
            viewer.layers.clear()
            add_layer_with_right_contrast(deconv_result, 'Deconv', viewer)
        self.progress_dialog.setValue(100)
        
    def apply_deconv(self):
        self.progress_dialog = QProgressDialog("Processing...", 'Cancel', 0, 100, self)
        self.progress_dialog.setWindowTitle('Applying')
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setValue(0)
        self.progress_dialog.show()

        voltage = self.voltage_input.value()
        cs = self.cs_input.value()
        defocus = self.defocus_input.value()
        pixel_size = self.pixel_size_input.value()
        snrfalloff = self.snrfalloff_input.value()
        deconvstrength = self.deconvstrength_input.value()
        highpassnyquist = self.highpassnyquist_input.value()

        self.progress_dialog.setValue(50)

        # 构造命令行
        star_file_path = self.tomo_viewer.tomo_path_and_stage.tomograms_star_path  # 假设 self.data 是 STAR 文件路径
        deconv_folder = os.path.dirname(self.tomo_viewer.tomo_path_and_stage.deconv_tomo_path)

        command = [
            "isonet.py", "deconv", star_file_path,
            "--deconv_folder", deconv_folder,
            "--voltage", str(voltage),
            "--cs", str(cs),
            "--snrfalloff", str(snrfalloff),
            "--deconvstrength", str(deconvstrength),
            "--highpassnyquist", str(highpassnyquist),
            "--ncpu", '4',
        ]

        # 执行命令行
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print("deconv_tomo output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error during deconv_tomo execution: {e.stderr}")

        # 读取deconv_tomo的结果
        deconv_result_path = self.tomo_viewer.tomo_path_and_stage.deconv_tomo_path
        if os.path.exists(deconv_result_path):
            with mrcfile.open(deconv_result_path, mode='r') as output_mrc:
                deconv_result = output_mrc.data

            add_layer_with_right_contrast(deconv_result, 'deconv_tomo', self.viewer)
            deconv_tomo_layer = self.viewer.layers['deconv_tomo']
            self.viewer.layers.move(self.viewer.layers.index(deconv_tomo_layer), 0)
            self.viewer.layers['ori_tomo'].visible = False
            self.progress_dialog.setValue(100)
            self.viewer.layers['edit vesicles'].data = None
            self.tomo_viewer.print("Finish Deconvolution.")
            self.tomo_viewer.show_current_state()
        else:
            print(f"Deconv result not found at: {deconv_result_path}")
            self.progress_dialog.setValue(100)

        # deconv_result = deconv_tomo(self.data, None, angpix=pixel_size, voltage=voltage, cs=cs, defocus=defocus, 
        #                             snrfalloff=snrfalloff, deconvstrength=deconvstrength, 
        #                             highpassnyquist=highpassnyquist, phaseflipped=False, phaseshift=0, ncpu=4)
        # # self.viewer.layers.clear()
        # directory = os.path.dirname(self.tomo_viewer.tomo_path_and_stage.deconv_tomo_path)
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        #     print(f"Created directory: {directory}")
        # with mrcfile.new(self.tomo_viewer.tomo_path_and_stage.deconv_tomo_path, overwrite=True) as output_mrc:
        #     output_mrc.set_data(deconv_result)
        #     output_mrc.voxel_size = 17.14
        # add_layer_with_right_contrast(deconv_result, 'deconv_tomo', self.viewer)
        # deconv_tomo_layer = self.viewer.layers['deconv_tomo']
        # self.viewer.layers.move(self.viewer.layers.index(deconv_tomo_layer), 0)
        # self.viewer.layers['ori_tomo'].visible = False
        # self.progress_dialog.setValue(100)
        # self.viewer.layers['edit vesicles'].data = None
        # self.tomo_viewer.print("Finish Deconvolution.")
        # self.tomo_viewer.show_current_state()
        self.close()
