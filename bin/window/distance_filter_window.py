import os
import shutil
import numpy as np
import xml.etree.ElementTree as ET
import mrcfile
from qtpy.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton
from qtpy.QtGui import QDoubleValidator

class DistanceFilterWindow(QDialog):
    def __init__(self, tomo_viewer):
        super().__init__(tomo_viewer.viewer.window.qt_viewer)
        self.tomo_viewer = tomo_viewer
        self.setWindowTitle("Filter Vesicles by Distance")
        self.setModal(True)
        self.resize(300, 150)
        
        layout = QVBoxLayout()
        
        # 使用 QHBoxLayout 来水平排列标签和输入框
        input_layout = QHBoxLayout()
        
        self.distance_label = QLabel("Distance (nm):")
        self.distance_input = QLineEdit()
        self.distance_input.setPlaceholderText("Enter distance in nm")
        self.distance_input.setValidator(QDoubleValidator(0.0, 1e6, 2))
        
        input_layout.addWidget(self.distance_label)
        input_layout.addWidget(self.distance_input)
        
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_filter)
        
        layout.addLayout(input_layout)  # 将水平布局添加到主布局
        layout.addWidget(self.apply_button)
        
        self.setLayout(layout)
    
    def apply_filter(self):
        """应用基于输入距离的囊泡过滤。"""
        try:
            distance_nm = float(self.distance_input.text())
            if distance_nm <= 0:
                raise ValueError("Distance must be positive.")
            self.accept()
            self.filter_vesicle(distance_nm)
        except ValueError as e:
            # 打印错误信息
            self.tomo_viewer.print(f"Invalid Input: {str(e)}")

    def get_pixel_size_from_xml(self, xml_path):
        """从 XML 文件中提取像素大小。"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            pixel_size = float(root.attrib.get('pixelSize', 1.714))  # 默认像素大小为 1.714
            return pixel_size
        except Exception as e:
            # 打印错误信息
            self.tomo_viewer.print(f"Failed to get pixel size from XML: {str(e)}")
            return 1.714

    def get_patch_around_point(self, data, z, y, x, size=128):
        """
        从 3D 数据中提取指定大小的补丁，处理边界，并用 0 填充缺失部分。

        参数：
            data (np.ndarray): 3D 图像数据 (z, y, x)。
            z, y, x (int): 中心点的坐标。
            size (int): 每个轴上的补丁大小。

        返回：
            patch (np.ndarray): 提取的补丁。
        """
        half_size = size // 2
        patch = np.zeros((size, size, size), dtype=data.dtype)
        
        # 定义补丁的起始和结束坐标
        z_min = z - half_size
        z_max = z + half_size
        y_min = y - half_size
        y_max = y + half_size
        x_min = x - half_size
        x_max = x + half_size

        # 图像数据的维度
        z_dim, y_dim, x_dim = data.shape

        # 计算补丁与图像边界的重叠部分，并确定在补丁中的位置
        # Z 轴
        if z_min < 0:
            patch_z_start = -z_min
            z_min = 0
        else:
            patch_z_start = 0
        if z_max > z_dim:
            patch_z_end = size - (z_max - z_dim)
            z_max = z_dim
        else:
            patch_z_end = z_max - (z - half_size)

        # Y 轴
        if y_min < 0:
            patch_y_start = -y_min
            y_min = 0
        else:
            patch_y_start = 0
        if y_max > y_dim:
            patch_y_end = size - (y_max - y_dim)
            y_max = y_dim
        else:
            patch_y_end = y_max - (y - half_size)

        # X 轴
        if x_min < 0:
            patch_x_start = -x_min
            x_min = 0
        else:
            patch_x_start = 0
        if x_max > x_dim:
            patch_x_end = size - (x_max - x_dim)
            x_max = x_dim
        else:
            patch_x_end = x_max - (x - half_size)

        # 计算图像数据中的有效区域
        data_z_start = z_min
        data_z_end = z_max
        data_y_start = y_min
        data_y_end = y_max
        data_x_start = x_min
        data_x_end = x_max

        # 计算补丁中的放置位置
        patch_z_end = patch_z_start + (data_z_end - data_z_start)
        patch_y_end = patch_y_start + (data_y_end - data_y_start)
        patch_x_end = patch_x_start + (data_x_end - data_x_start)

        # 将有效数据填充到补丁中
        patch[patch_z_start:patch_z_end, patch_y_start:patch_y_end, patch_x_start:patch_x_end] = \
            data[data_z_start:data_z_end, data_y_start:data_y_end, data_x_start:data_x_end]
        
        return patch

    def filter_vesicle(self, distance_nm):
        """基于距离过滤囊泡并更新标签数据，同时剪裁并保存子图像。"""
        try:
            # 读取标签数据（假设为 3D numpy 数组）
            label_data = self.tomo_viewer.viewer.layers['label'].data
            
            # 获取 XML 文件中的像素大小
            pixel_size = self.get_pixel_size_from_xml(self.tomo_viewer.tomo_path_and_stage.ori_xml_path)
            
            # 解析 XML 数据
            tree = ET.parse(self.tomo_viewer.tomo_path_and_stage.ori_xml_path)
            root = tree.getroot()
            
            # 筛选距离小于指定距离的囊泡，并将其类型改为 'others'
            for vesicle in root.findall('Vesicle'):
                distance = float(vesicle.find('Distance').attrib['d'])
                if distance * pixel_size < distance_nm:
                    # 将类型改为 'others'
                    type_elem = vesicle.find('Type')
                    if type_elem is not None and type_elem.attrib.get('t') == 'vesicle':
                        type_elem.set('t', 'others')
            
            # 保存修改后的 XML
            tree.write(self.tomo_viewer.tomo_path_and_stage.filter_xml_path, encoding='utf-8', xml_declaration=True)
            
            # 重新解析修改后的 XML
            tree_new = ET.parse(self.tomo_viewer.tomo_path_and_stage.filter_xml_path)
            root_new = tree_new.getroot()
            
            # 提取所有类型为 'others' 的 Vesicle 的 Center 坐标和 ID
            centers = []
            vesicle_ids = []
            for vesicle in root_new.findall('Vesicle'):
                vesicle_type = vesicle.find('Type').attrib.get('t')
                if vesicle_type == 'others':
                    center = vesicle.find('Center')
                    x = float(center.attrib['X'])
                    y = float(center.attrib['Y'])
                    z = float(center.attrib['Z'])
                    vesicle_id = vesicle.attrib.get('vesicleId')
                    centers.append((z, y, x))  # 假设 label_data 的顺序为 (z, y, x)
                    vesicle_ids.append(vesicle_id)
            
            # 根据 Center 坐标找到对应的 label 值
            labels_to_keep = set()
            for z_coord, y_coord, x_coord in centers:
                # 将物理坐标转换为体素索引
                z_idx = int(z_coord)
                y_idx = int(y_coord)
                x_idx = int(x_coord)
                
                # 检查索引是否在 label_data 的范围内
                if (0 <= z_idx < label_data.shape[0] and
                    0 <= y_idx < label_data.shape[1] and
                    0 <= x_idx < label_data.shape[2]):
                    label = label_data[z_idx, y_idx, x_idx]
                    if label != 0:
                        labels_to_keep.add(label)
            
            # 更新 label_data，仅保留对应的 labels
            new_label_data = np.where(np.isin(label_data, list(labels_to_keep)), label_data, 0)
            
            # 添加更新后的标签到查看器
            self.tomo_viewer.viewer.layers['label'].visible = False
            self.tomo_viewer.viewer.add_labels(new_label_data, name="filter_labels")
            self.tomo_viewer.viewer.layers["filter_labels"].opacity = 0.5
            
            # 打印成功信息
            self.tomo_viewer.print(f"Filtering completed, new XML saved at {self.tomo_viewer.tomo_path_and_stage.filter_xml_path}")
            
            # 开始剪裁和保存子图像
            self.extract_and_save_patches(centers, vesicle_ids)
        
        except Exception as e:
            # 打印错误信息
            self.tomo_viewer.print(f"Vesicle filtering failed: {str(e)}")
    
    def extract_and_save_patches(self, centers, vesicle_ids, size=128):
        """
        根据中心点坐标剪裁图像并保存为 .mrc 文件。

        参数：
            centers (list of tuples): 每个元组包含 (z, y, x) 坐标。
            vesicle_ids (list): 每个囊泡的 ID。
            size (int): 补丁大小。
        """
        try:
            # 获取图像数据
            corrected_tomo_layer = self.tomo_viewer.viewer.layers['corrected_tomo']
            if corrected_tomo_layer is None:
                self.tomo_viewer.print("Layer 'corrected_tomo' not found.")
                return
            tomo_data = corrected_tomo_layer.data  # 假设为 3D numpy 数组

            # 确定保存路径
            filter_xml_dir = os.path.dirname(self.tomo_viewer.tomo_path_and_stage.filter_xml_path)
            extract_dir = os.path.join(filter_xml_dir, "extractRRP_3D")
            
            # 如果目录存在，删除它
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            
            # 创建新的目录
            os.makedirs(extract_dir, exist_ok=True)
            
            # 获取像素大小（假设与 get_pixel_size_from_xml 相同）
            pixel_size = self.get_pixel_size_from_xml(self.tomo_viewer.tomo_path_and_stage.ori_xml_path)
            voxel_size = (pixel_size, pixel_size, pixel_size)  # 假设各轴相同
            
            # 遍历所有中心点，剪裁并保存
            for center, vesicle_id in zip(centers, vesicle_ids):
                z, y, x = center
                z_idx = int(z)
                y_idx = int(y)
                x_idx = int(x)
                
                # 提取补丁
                patch = self.get_patch_around_point(tomo_data, z_idx, y_idx, x_idx, size=size)
                
                # 定义输出路径
                output_filename = f"vesicle_{vesicle_id}.mrc"
                output_path = os.path.join(extract_dir, output_filename)
                
                # 保存为 .mrc 文件
                with mrcfile.new(output_path, overwrite=True) as output_mrc:
                    output_mrc.set_data(patch.astype(np.float32))
                    output_mrc.voxel_size = voxel_size
            
            # 打印成功信息
            self.tomo_viewer.print(f"Patches extracted and saved to {extract_dir}")
        
        except Exception as e:
            # 打印错误信息
            self.tomo_viewer.print(f"Failed to extract and save patches: {str(e)}")
