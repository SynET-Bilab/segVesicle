import xml.etree.ElementTree as ET
import numpy as np
import os
from qtpy.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QComboBox, QMessageBox
)
from qtpy.QtCore import Qt
import napari

class VesicleAnnotationWindow(QMainWindow):
    def __init__(self, parent):
        super().__init__(parent.viewer.window.qt_viewer)
        self.setWindowTitle('Annotate Vesicle Types')
        self.parent = parent
        self.viewer = parent.viewer
        self.current_index = 0  # 当前起始囊泡索引

        # 从XML文件中读取囊泡信息
        self.vesicles = self.load_vesicles_from_xml()
        
        # 初始化类别字典，保存每个囊泡的类别
        self.classes = {}
        for vesicle in self.vesicles:
            vesicle_id = vesicle.attrib['vesicleId']
            type_element = vesicle.find('Type')
            if type_element is not None:
                self.classes[vesicle_id] = type_element.attrib.get('t', 'others')
            else:
                self.classes[vesicle_id] = 'others'  # 默认类别为'others'

        # 创建12个Napari Viewer用于显示图像小块
        self.patch_viewers = [napari.Viewer(show=False) for _ in range(12)]
        self.patch_layers = [None] * 12

        # 保存每个类别输入框，以便后续操作
        self.class_inputs = [None] * 11  # 前11个对应类别输入框

        # 创建界面元素
        self.init_ui()

        # 显示当前的囊泡组
        self.display_current_vesicles()

    def init_ui(self):
        # 主部件
        main_widget = QWidget()
        main_layout = QGridLayout()

        # 定义类别选项
        self.class_options = ['false', 'tether', 'contact', 'omega', 'pit', 'CCV', 'endosome', 'DCV', 'others']

        # 创建3x4的网格布局
        for i in range(12):
            row = i // 4
            col = i % 4
            if i == 11:  # 最后一个网格放置左右按钮
                button_layout = QHBoxLayout()
                self.prev_button = QPushButton("Previous")
                self.prev_button.clicked.connect(self.prev_vesicles)
                self.next_button = QPushButton("Next")
                self.next_button.clicked.connect(self.next_vesicles)
                button_layout.addWidget(self.prev_button)
                button_layout.addWidget(self.next_button)
                container = QWidget()
                container.setLayout(button_layout)
                main_layout.addWidget(container, row, col)
            else:
                container = QWidget()
                container_layout = QVBoxLayout()

                # 添加Napari Viewer
                container_layout.addWidget(self.patch_viewers[i].window.qt_viewer)

                # 添加类别选择框
                vesicle_class_input = QComboBox()
                vesicle_class_input.addItems(self.class_options)
                vesicle_class_input.setCurrentText('others')  # 默认选择 'others'
                # 设置对象名称，方便后续查找
                vesicle_class_input.setObjectName(f'class_input_{i}')
                vesicle_class_input.currentTextChanged.connect(lambda val, idx=i: self.update_class(idx, val))
                container_layout.addWidget(vesicle_class_input)

                container.setLayout(container_layout)
                main_layout.addWidget(container, row, col)
                self.class_inputs[i] = vesicle_class_input  # 保存输入框引用

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def load_vesicles_from_xml(self):
        # 首先检查是否存在class_xml_path
        if hasattr(self.parent.tomo_path_and_stage, 'class_xml_path') and \
           os.path.exists(self.parent.tomo_path_and_stage.class_xml_path):
            xml_path = self.parent.tomo_path_and_stage.class_xml_path
        else:
            # 如果class_xml_path不存在，使用filter_xml_path路径
            xml_path = self.parent.tomo_path_and_stage.filter_xml_path

        self.xml_path = xml_path  # 保存当前使用的XML路径

        # 读取XML文件
        tree = ET.parse(xml_path)
        root = tree.getroot()
        vesicles = root.findall('Vesicle')

        # 过滤Type t不等于"vesicle"的囊泡
        target_vesicles = [v for v in vesicles if v.find('Type') is not None and v.find('Type').attrib.get('t') != 'vesicle']

        self.tree = tree
        self.root = root
        self.all_vesicles = vesicles  # 所有囊泡，包括已标记和未标记

        return target_vesicles

    def display_current_vesicles(self):
        total_vesicles = len(self.vesicles)
        for i in range(11):  # 前11个用于显示囊泡
            idx = self.current_index + i
            if idx < total_vesicles:
                vesicle = self.vesicles[idx]
                center = vesicle.find('Center')
                x = float(center.attrib['X'])
                y = float(center.attrib['Y'])
                z = float(center.attrib['Z'])
                vesicle_id = vesicle.attrib['vesicleId']

                # 更新类别选择框的值
                class_value = self.classes[vesicle_id]
                vesicle_class_input = self.class_inputs[i]
                if vesicle_class_input:
                    vesicle_class_input.setEnabled(True)
                    if class_value in self.class_options:
                        vesicle_class_input.setCurrentText(class_value)
                    else:
                        vesicle_class_input.setCurrentText('others')  # 默认值

                patch = self.get_patch_around_point(z, y, x, size=128)

                # 调整灰度级
                min_val = np.percentile(patch, 0.1)
                max_val = np.percentile(patch, 99)

                # 更新Napari Viewer
                if self.patch_layers[i]:
                    self.patch_viewers[i].layers.remove(self.patch_layers[i])
                self.patch_layers[i] = self.patch_viewers[i].add_image(
                    patch, name=f'Vesicle {vesicle_id}',
                    contrast_limits=[min_val, max_val],
                    opacity=0.8,
                    gamma=0.75
                )
            else:
                # 如果没有更多的囊泡，清空Viewer并禁用选择框
                if self.patch_layers[i]:
                    self.patch_viewers[i].layers.clear()
                    self.patch_layers[i] = None
                vesicle_class_input = self.class_inputs[i]
                if vesicle_class_input:
                    vesicle_class_input.setEnabled(False)
                    vesicle_class_input.setCurrentText('others')  # 重置为默认值

    def get_patch_around_point(self, z, y, x, size=128):
        half_size = size // 2
        data = self.viewer.layers[0].data  # 假设图像数据在第一个层
        z, y, x = int(round(z)), int(round(y)), int(round(x))
        
        # 定义裁剪区域的起始和结束坐标
        z_min = z - half_size
        z_max = z + half_size
        y_min = y - half_size
        y_max = y + half_size
        x_min = x - half_size
        x_max = x + half_size

        # 初始化填充后的patch，默认值为0（可根据需要修改）
        patch = np.zeros((size, size, size), dtype=data.dtype)
        
        # 计算图像数据的维度
        z_dim, y_dim, x_dim = data.shape
        
        # 计算裁剪区域与图像边界的重叠部分
        # 对Z轴
        if z_min < 0:
            patch_z_start = -z_min
            z_min = 0
        else:
            patch_z_start = 0
        if z_max > z_dim:
            patch_z_end = size - (z_max - z_dim)
            z_max = z_dim
        else:
            patch_z_end = z_max - (z + half_size)

        # 对Y轴
        if y_min < 0:
            patch_y_start = -y_min
            y_min = 0
        else:
            patch_y_start = 0
        if y_max > y_dim:
            patch_y_end = size - (y_max - y_dim)
            y_max = y_dim
        else:
            patch_y_end = y_max - (y + half_size)

        # 对X轴
        if x_min < 0:
            patch_x_start = -x_min
            x_min = 0
        else:
            patch_x_start = 0
        if x_max > x_dim:
            patch_x_end = size - (x_max - x_dim)
            x_max = x_dim
        else:
            patch_x_end = x_max - (x + half_size)

        # 计算在图像数据中的有效区域
        data_z_start = z_min
        data_z_end = z_max
        data_y_start = y_min
        data_y_end = y_max
        data_x_start = x_min
        data_x_end = x_max

        # 计算在patch中的放置位置
        patch_z_start = max(patch_z_start, 0)
        patch_z_end = patch_z_start + (data_z_end - data_z_start)
        patch_y_start = max(patch_y_start, 0)
        patch_y_end = patch_y_start + (data_y_end - data_y_start)
        patch_x_start = max(patch_x_start, 0)
        patch_x_end = patch_x_start + (data_x_end - data_x_start)

        # 将有效数据填充到patch中
        patch[patch_z_start:patch_z_end, patch_y_start:patch_y_end, patch_x_start:patch_x_end] = \
            data[data_z_start:data_z_end, data_y_start:data_y_end, data_x_start:data_x_end]
        
        return patch
        # half_size = size // 2
        # data = self.viewer.layers[0].data  # 假设图像数据在第一个层
        # z, y, x = int(round(z)), int(round(y)), int(round(x))
        # z_min = max(z - half_size, 0)
        # z_max = min(z + half_size, data.shape[0])
        # y_min = max(y - half_size, 0)
        # y_max = min(y + half_size, data.shape[1])
        # x_min = max(x - half_size, 0)
        # x_max = min(x + half_size, data.shape[2])
        # patch = data[z_min:z_max, y_min:y_max, x_min:x_max]
        # return patch

    def update_class(self, idx, value):
        # 更新类别字典
        vesicle_idx = self.current_index + idx
        if vesicle_idx < len(self.vesicles):
            vesicle_id = self.vesicles[vesicle_idx].attrib['vesicleId']
            self.classes[vesicle_id] = value

    def prev_vesicles(self):
        self.save_current_classes()
        if self.current_index >= 11:
            self.current_index -= 11
            self.display_current_vesicles()
        else:
            QMessageBox.information(self, "Warning", "This is the first page.")

    def next_vesicles(self):
        self.save_current_classes()
        if self.current_index + 11 < len(self.vesicles):
            self.current_index += 11
            self.display_current_vesicles()
        else:
            QMessageBox.information(self, "Congratulation!", "You have finished annotations!")

    def save_current_classes(self):
        for i in range(11):
            idx = self.current_index + i
            if idx < len(self.vesicles):
                vesicle = self.vesicles[idx]
                vesicle_id = vesicle.attrib['vesicleId']
                class_value = self.classes[vesicle_id]
                # 更新XML中的Type t属性
                type_element = vesicle.find('Type')
                if type_element is not None:
                    type_element.set('t', class_value)
                else:
                    # 如果没有Type元素，则创建一个
                    type_element = ET.SubElement(vesicle, 'Type')
                    type_element.set('t', class_value)

        # 保存XML文件到filter_xml_path或指定路径
        self.tree.write(self.parent.tomo_path_and_stage.class_xml_path, encoding='utf-8', xml_declaration=False)
        # 确保格式正确
        with open(self.parent.tomo_path_and_stage.class_xml_path, 'r+', encoding='utf-8') as file:
            content = file.read()
            file.seek(0)
            # 手动添加换行符
            file.write(content.replace('"><', '">\n<'))

    def closeEvent(self, event):
        # 窗口关闭时，保存类别信息并更新mask
        self.save_current_classes()
        # self.update_filter_labels()  # 如果需要更新mask，可以在这里调用
        super().closeEvent(event)
