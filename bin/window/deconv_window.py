import napari
import numpy as np
from qtpy.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit, QGridLayout, QDoubleSpinBox, QProgressDialog
from qtpy.QtCore import Qt

from util.add_layer_with_right_contrast import add_layer_with_right_contrast
from util.deconvolution import deconv_tomo

# class DeconvWindow(QMainWindow):
#     def __init__(self, viewer: napari.Viewer):
        
#         super().__init__()
#         self.setWindowTitle('Deconv Preview')
#         # self.setStyleSheet("background-color: gray;")
#         # self.showMaximized()
        
#         self.viewer = viewer
#         self.viewer_left = napari.Viewer(show=False)
#         self.viewer_right = napari.Viewer(show=False)
    
#         self.points = viewer.layers['edit vesicles'].data
#         self.data = viewer.layers['ori_tomo'].data
        
#         self.crop_data = self.get_cropped_data()
#         add_layer_with_right_contrast(self.crop_data, 'Ori', self.viewer_left)
#         # self.viewer_left.add_image(self.crop_data, name='Ori')
        
#         # Default values
#         self.default_values = {
#             'voltage': 300.0,
#             'cs': 2.7,
#             'defocus': 0.0,
#             'pixel_size': 17.14,
#             'snrfalloff': 1.0,
#             'deconvstrength': 1.0,
#             'highpassnyquist': 0.02
#         }

#         # Deconv parameter inputs
#         self.voltage_input = QDoubleSpinBox(self)
#         self.voltage_input.setRange(100.0, 1000.0)
#         self.voltage_input.setValue(self.default_values['voltage'])

#         self.cs_input = QDoubleSpinBox(self)
#         self.cs_input.setRange(0.0, 5.0)
#         self.cs_input.setValue(self.default_values['cs'])

#         self.defocus_input = QDoubleSpinBox(self)
#         self.defocus_input.setRange(0.0, 5.0)
#         self.defocus_input.setValue(self.default_values['defocus'])

#         self.pixel_size_input = QDoubleSpinBox(self)
#         self.pixel_size_input.setRange(0.1, 100.0)
#         self.pixel_size_input.setValue(self.default_values['pixel_size'])

#         self.snrfalloff_input = QDoubleSpinBox(self)
#         self.snrfalloff_input.setRange(0.0, 10.0)
#         self.snrfalloff_input.setValue(self.default_values['snrfalloff'])

#         self.deconvstrength_input = QDoubleSpinBox(self)
#         self.deconvstrength_input.setRange(0.0, 10.0)
#         self.deconvstrength_input.setValue(self.default_values['deconvstrength'])

#         self.highpassnyquist_input = QDoubleSpinBox(self)
#         self.highpassnyquist_input.setRange(0.0, 1.0)
#         self.highpassnyquist_input.setValue(self.default_values['highpassnyquist'])

#         self.apply_button = QPushButton('Apply Deconv', self)
#         self.apply_button.clicked.connect(self.apply_deconv)
        
#         self.preview_button = QPushButton('Preview Deconv', self)
#         self.preview_button.clicked.connect(self.preview_deconv)

#         layout = QGridLayout()
#         layout.addWidget(self.viewer_left.window.qt_viewer, 0, 0, 1, 1)
        
#         mid_layout = QVBoxLayout()

#         self.add_parameter(mid_layout, 'Voltage (kV):', self.voltage_input)
#         self.add_parameter(mid_layout, 'Cs (mm):', self.cs_input)
#         self.add_parameter(mid_layout, 'Defocus (um):', self.defocus_input)
#         self.add_parameter(mid_layout, 'Pixel Size (A):', self.pixel_size_input)
#         self.add_parameter(mid_layout, 'SNR Falloff:', self.snrfalloff_input)
#         self.add_parameter(mid_layout, 'Deconv Strength:', self.deconvstrength_input)
#         self.add_parameter(mid_layout, 'Highpass Nyquist:', self.highpassnyquist_input)
        
#         # button_layout = QHBoxLayout()
#         mid_layout.addWidget(self.preview_button)
#         mid_layout.addWidget(self.apply_button)
#         layout.addLayout(mid_layout, 0, 1)

#         layout.addWidget(self.viewer_right.window.qt_viewer, 0, 2, 1, 1)

#         container = QWidget()
#         container.setLayout(layout)
#         self.setCentralWidget(container)
    
#     def add_parameter(self, layout, label_text, widget):
#         h_layout = QHBoxLayout()
#         h_layout.addWidget(QLabel(label_text))
#         h_layout.addWidget(widget)
#         layout.addLayout(h_layout)
    
#     def get_cropped_data(self):
#         # 计算剪裁区域
#         p1 = self.points[0]  # 点的格式是 (z, y, x)
#         p2 = self.points[1]
#         z_val = int(round(p1[0]))  # 两个点的 Z 值相同
#         min_y, min_x = np.min([p1[1:], p2[1:]], axis=0)
#         max_y, max_x = np.max([p1[1:], p2[1:]], axis=0)
#         # 确保坐标是整数
#         min_y, min_x = int(min_y), int(min_x)
#         max_y, max_x = int(max_y), int(max_x)
#         # 确定 Z 轴上的剪裁范围
#         min_z = max(0, z_val - 50)
#         max_z = min(self.data.shape[0], z_val + 50)
#         z_size = self.data.shape[0]
#         # 确定 Z 轴上的剪裁范围
#         min_z = max(0, z_val - 25)
#         max_z = min(z_size, z_val + 25 + 1)  # 加1以确保包含第z_val + 50切片
#         # 剪裁图像
#         cropped_image = self.data[min_z:max_z, min_y:max_y + 1, min_x:max_x + 1]
#         return cropped_image
    
#     def preview_deconv(self):
#         # Read input parameters
#         voltage = self.voltage_input.value()
#         cs = self.cs_input.value()
#         defocus = self.defocus_input.value()
#         pixel_size = self.pixel_size_input.value()
#         snrfalloff = self.snrfalloff_input.value()
#         deconvstrength = self.deconvstrength_input.value()
#         highpassnyquist = self.highpassnyquist_input.value()

#         # Call the deconvolution function with the given parameters
#         deconv_result = deconv_tomo(self.crop_data, None, angpix=pixel_size, voltage=voltage, cs=cs, defocus=defocus, 
#                                         snrfalloff=snrfalloff, deconvstrength=deconvstrength, 
#                                         highpassnyquist=highpassnyquist, phaseflipped=False, phaseshift=0, ncpu=4)
#         # Clear all layers in the right viewer
#         self.viewer_right.layers.clear()
        
#         # Display the result in the right viewer
#         add_layer_with_right_contrast(deconv_result, 'Deconv', self.viewer_right)
        
#     def apply_deconv(self):
#         self.progress_dialog = QProgressDialog("Processing...", 'Cancel', 0, 100, self)
#         self.progress_dialog.setWindowTitle('Applying')
#         self.progress_dialog.setWindowModality(Qt.WindowModal)
#         self.progress_dialog.setValue(0)
#         self.progress_dialog.show()
#         # Read input parameters
#         voltage = self.voltage_input.value()
#         cs = self.cs_input.value()
#         defocus = self.defocus_input.value()
#         pixel_size = self.pixel_size_input.value()
#         snrfalloff = self.snrfalloff_input.value()
#         deconvstrength = self.deconvstrength_input.value()
#         highpassnyquist = self.highpassnyquist_input.value()
#         self.progress_dialog.setValue(50)
#         # Call the deconvolution function with the given parameters
#         deconv_result = deconv_tomo(self.data, None, angpix=pixel_size, voltage=voltage, cs=cs, defocus=defocus, 
#                                         snrfalloff=snrfalloff, deconvstrength=deconvstrength, 
#                                         highpassnyquist=highpassnyquist, phaseflipped=False, phaseshift=0, ncpu=4)
#         # Clear all layers in the right viewer
#         self.viewer_right.layers.clear()
        
#         # Display the result in the right viewer
#         add_layer_with_right_contrast(deconv_result, 'deconv_tomo', self.viewer)
#         deconv_tomo_layer = self.viewer.layers['deconv_tomo']
#         self.viewer.layers.move(self.viewer.layers.index(deconv_tomo_layer), 0)
#         self.viewer.layers['ori_tomo'].visible = False
#         self.progress_dialog.setValue(100)
#         self.close()

"""
class DeconvWindow(QMainWindow):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.setWindowTitle('Deconv Preview')
        
        self.viewer = viewer
        self.preview_viewers = [napari.Viewer(show=False) for _ in range(6)]

        self.points = viewer.layers['edit vesicles'].data
        self.data = viewer.layers['ori_tomo'].data
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

        self.cs_input = QDoubleSpinBox(self)
        self.cs_input.setRange(0.0, 5.0)
        self.cs_input.setValue(self.default_values['cs'])

        self.defocus_input = QDoubleSpinBox(self)
        self.defocus_input.setRange(0.0, 5.0)
        self.defocus_input.setValue(self.default_values['defocus'])

        self.pixel_size_input = QDoubleSpinBox(self)
        self.pixel_size_input.setRange(0.1, 100.0)
        self.pixel_size_input.setValue(self.default_values['pixel_size'])

        self.snrfalloff_input = QDoubleSpinBox(self)
        self.snrfalloff_input.setRange(0.0, 10.0)
        self.snrfalloff_input.setValue(self.default_values['snrfalloff'])

        self.deconvstrength_input = QDoubleSpinBox(self)
        self.deconvstrength_input.setRange(0.0, 10.0)
        self.deconvstrength_input.setValue(self.default_values['deconvstrength'])

        self.highpassnyquist_input = QDoubleSpinBox(self)
        self.highpassnyquist_input.setRange(0.0, 1.0)
        self.highpassnyquist_input.setValue(self.default_values['highpassnyquist'])

        self.apply_button = QPushButton('Apply Deconv', self)
        self.apply_button.clicked.connect(self.apply_deconv)
        
        self.preview_button = QPushButton('Preview Deconv', self)
        self.preview_button.clicked.connect(self.preview_deconv)

        layout = QGridLayout()

        # Add 2x3 grid of preview viewers
        snrfalloff_values = [0.3 + 0.2 * i for i in range(6)]
        for i, (viewer, snrfalloff) in enumerate(zip(self.preview_viewers, snrfalloff_values)):
            row = (i // 3) * 2  # 每个预览图占据两行
            col = i % 3

            # 创建一个包含 VispyCanvas 的 QWidget 包装器
            container = QWidget()
            container_layout = QVBoxLayout()
            container_layout.addWidget(viewer.window.qt_viewer)
            container.setLayout(container_layout)
            container.setFixedHeight(600)
            container.setFixedWidth(600)
            
            layout.addWidget(container, row, col)
            layout.addWidget(QLabel(f'SNR Falloff: {snrfalloff}'), row + 1, col)
        
        # Right side layout for parameter inputs
        right_layout = QVBoxLayout()
        self.add_parameter(right_layout, 'Voltage (kV):', self.voltage_input)
        self.add_parameter(right_layout, 'Cs (mm):', self.cs_input)
        self.add_parameter(right_layout, 'Defocus (um):', self.defocus_input)
        self.add_parameter(right_layout, 'Pixel Size (A):', self.pixel_size_input)
        self.add_parameter(right_layout, 'SNR Falloff:', self.snrfalloff_input)
        self.add_parameter(right_layout, 'Deconv Strength:', self.deconvstrength_input)
        self.add_parameter(right_layout, 'Highpass Nyquist:', self.highpassnyquist_input)
        right_layout.addWidget(self.preview_button)
        right_layout.addWidget(self.apply_button)
        
        layout.addLayout(right_layout, 0, 3, 6, 1)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
    
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
    
    def preview_deconv(self):
        voltage = self.voltage_input.value()
        cs = self.cs_input.value()
        defocus = self.defocus_input.value()
        pixel_size = self.pixel_size_input.value()
        deconvstrength = self.deconvstrength_input.value()
        highpassnyquist = self.highpassnyquist_input.value()

        snrfalloff_values = [0.3 + 0.2 * i for i in range(6)]
        for viewer, snrfalloff in zip(self.preview_viewers, snrfalloff_values):
            deconv_result = deconv_tomo(self.crop_data, None, angpix=pixel_size, voltage=voltage, cs=cs, defocus=defocus, 
                                        snrfalloff=snrfalloff, deconvstrength=deconvstrength, 
                                        highpassnyquist=highpassnyquist, phaseflipped=False, phaseshift=0, ncpu=4)
            viewer.layers.clear()
            add_layer_with_right_contrast(deconv_result, 'Deconv', viewer)
        
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

        deconv_result = deconv_tomo(self.data, None, angpix=pixel_size, voltage=voltage, cs=cs, defocus=defocus, 
                                    snrfalloff=snrfalloff, deconvstrength=deconvstrength, 
                                    highpassnyquist=highpassnyquist, phaseflipped=False, phaseshift=0, ncpu=4)
        self.viewer.layers.clear()
        add_layer_with_right_contrast(deconv_result, 'deconv_tomo', self.viewer)
        deconv_tomo_layer = self.viewer.layers['deconv_tomo']
        self.viewer.layers.move(self.viewer.layers.index(deconv_tomo_layer), 0)
        self.viewer.layers['ori_tomo'].visible = False
        self.progress_dialog.setValue(100)
        self.close()
"""
class DeconvWindow(QMainWindow):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.setWindowTitle('Deconv Preview')
        
        self.viewer = viewer
        self.preview_viewers = [napari.Viewer(show=False) for _ in range(11)]

        self.points = viewer.layers['edit vesicles'].data
        self.data = viewer.layers['ori_tomo'].data
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
    
    def preview_deconv(self):
        voltage = self.voltage_input.value()
        cs = self.cs_input.value()
        defocus = self.defocus_input.value()
        pixel_size = self.pixel_size_input.value()
        deconvstrength = self.deconvstrength_input.value()
        highpassnyquist = self.highpassnyquist_input.value()

        snrfalloff_values = [0.3 + 0.1 * i for i in range(11)]
        for viewer, snrfalloff in zip(self.preview_viewers, snrfalloff_values):
            deconv_result = deconv_tomo(self.crop_data, None, angpix=pixel_size, voltage=voltage, cs=cs, defocus=defocus, 
                                        snrfalloff=snrfalloff, deconvstrength=deconvstrength, 
                                        highpassnyquist=highpassnyquist, phaseflipped=False, phaseshift=0, ncpu=4)
            viewer.layers.clear()
            add_layer_with_right_contrast(deconv_result, 'Deconv', viewer)
        
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

        deconv_result = deconv_tomo(self.data, None, angpix=pixel_size, voltage=voltage, cs=cs, defocus=defocus, 
                                    snrfalloff=snrfalloff, deconvstrength=deconvstrength, 
                                    highpassnyquist=highpassnyquist, phaseflipped=False, phaseshift=0, ncpu=4)
        # self.viewer.layers.clear()
        add_layer_with_right_contrast(deconv_result, 'deconv_tomo', self.viewer)
        deconv_tomo_layer = self.viewer.layers['deconv_tomo']
        self.viewer.layers.move(self.viewer.layers.index(deconv_tomo_layer), 0)
        self.viewer.layers['ori_tomo'].visible = False
        self.progress_dialog.setValue(100)
        self.close()
