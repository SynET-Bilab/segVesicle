# finetune_model_window.py

import os
import json
import numpy as np
import mrcfile
from skimage.measure import regionprops, label
from PyQt5 import QtWidgets, QtCore

# 假设 ensure_model_exists 和 train_model 函数在同一目录下的 utils.py 文件中定义
from train.finetune import train_model
from util.model_exists import ensure_model_exists

class FinetuneModelWindow(QtWidgets.QDialog):
    def __init__(self, parent=None, tomo_path_and_stage=None, print_callback=None):
        super(FinetuneModelWindow, self).__init__(parent)
        self.setWindowTitle("Fine-tune Model")
        self.tomo_path_and_stage = tomo_path_and_stage
        self.print_callback = print_callback or print

        # 设置默认值
        self.default_batch_size = 8
        self.default_epochs = 10
        self.default_iso_model_pth = ensure_model_exists('vesicle_seg_model_1.h5')
        self.default_dec_model_pth = ensure_model_exists('vesicle_seg_model_2.h5')

        # 创建布局
        layout = QtWidgets.QVBoxLayout()

        # Batch Size
        self.batch_size_label = QtWidgets.QLabel("Batch Size:")
        self.batch_size_input = QtWidgets.QSpinBox()
        self.batch_size_input.setRange(1, 1024)
        self.batch_size_input.setValue(self.default_batch_size)
        layout.addWidget(self.batch_size_label)
        layout.addWidget(self.batch_size_input)

        # Epochs
        self.epochs_label = QtWidgets.QLabel("Epochs:")
        self.epochs_input = QtWidgets.QSpinBox()
        self.epochs_input.setRange(1, 1000)
        self.epochs_input.setValue(self.default_epochs)
        layout.addWidget(self.epochs_label)
        layout.addWidget(self.epochs_input)

        # ISO Model Path
        self.iso_model_label = QtWidgets.QLabel("ISO Model Path:")
        self.iso_model_input = QtWidgets.QLineEdit()
        self.iso_model_input.setText(self.default_iso_model_pth)
        self.iso_model_browse = QtWidgets.QPushButton("Browse")
        self.iso_model_browse.clicked.connect(self.browse_iso_model)
        iso_layout = QtWidgets.QHBoxLayout()
        iso_layout.addWidget(self.iso_model_input)
        iso_layout.addWidget(self.iso_model_browse)
        layout.addWidget(self.iso_model_label)
        layout.addLayout(iso_layout)

        # DEC Model Path
        self.dec_model_label = QtWidgets.QLabel("DEC Model Path:")
        self.dec_model_input = QtWidgets.QLineEdit()
        self.dec_model_input.setText(self.default_dec_model_pth)
        self.dec_model_browse = QtWidgets.QPushButton("Browse")
        self.dec_model_browse.clicked.connect(self.browse_dec_model)
        dec_layout = QtWidgets.QHBoxLayout()
        dec_layout.addWidget(self.dec_model_input)
        dec_layout.addWidget(self.dec_model_browse)
        layout.addWidget(self.dec_model_label)
        layout.addLayout(dec_layout)

        # Apply Button
        self.apply_button = QtWidgets.QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_finetune)
        layout.addWidget(self.apply_button)

        self.setLayout(layout)

    def browse_iso_model(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select ISO Model File", "", "Model Files (*.h5 *.pth)")
        if file_path:
            self.iso_model_input.setText(file_path)

    def browse_dec_model(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select DEC Model File", "", "Model Files (*.h5 *.pth)")
        if file_path:
            self.dec_model_input.setText(file_path)

    def apply_finetune(self):
        batch_size = self.batch_size_input.value()
        epochs = self.epochs_input.value()
        iso_model_pth = self.iso_model_input.text()
        dec_model_pth = self.dec_model_input.text()

        # 执行 finetune_model
        self.finetune_model(batch_size, epochs, iso_model_pth, dec_model_pth)
        
        # 关闭窗口
        self.accept()

    def finetune_model(self, batch_size, epochs, iso_model_pth, dec_model_pth):
        try:
            if os.path.exists(self.tomo_path_and_stage.new_label_file_path):
                os.system(f'cp {self.tomo_path_and_stage.new_label_file_path} {self.tomo_path_and_stage.label_path}')

            if os.path.exists(self.tomo_path_and_stage.new_json_file_path):
                os.system(f'cp {self.tomo_path_and_stage.new_json_file_path} {self.tomo_path_and_stage.json_file_path}')

            # 获取当前工作目录
            current_path = self.tomo_path_and_stage.current_path
            self.print_callback(f"当前工作目录: {current_path}")

            # 读取 segVesicle_QCheckBox_state.json 文件
            json_path = os.path.join(current_path, "segVesicle_QCheckBox_state.json")
            if not os.path.exists(json_path):
                self.print_callback(f"配置文件不存在: {json_path}")
                return

            with open(json_path, 'r') as f:
                seg_vesicle_state = json.load(f)
            self.print_callback(f"读取到的配置: {seg_vesicle_state}")

            # 获取所有为 true 的 tomo_name 并设置 base_tomo_name
            active_tomos = [tomo_name for tomo_name, state in seg_vesicle_state.items() if state]
            if not active_tomos:
                self.print_callback("没有需要处理的 tomo。")
                return

            self.print_callback(f"需要处理的 tomo 列表: {active_tomos}")

            processed_tomos = []
            base_tomo_names = []

            for tomo_name in active_tomos:
                if '-1' in tomo_name:
                    base_tomo_name = tomo_name.split('-1')[0]
                else:
                    base_tomo_name = tomo_name
                base_tomo_names.append(base_tomo_name)
                processed_tomos.append((tomo_name, base_tomo_name))

            # 获取每个 tomo 的三个关键路径
            tomo_paths = []
            for tomo_name, base_tomo_name in processed_tomos:
                deconv_tomo_path = os.path.join(current_path, tomo_name, 'ves_seg', 'tomo_deconv', f"{base_tomo_name}_wbp_resample.mrc")
                isonet_tomo_path = os.path.join(current_path, tomo_name, 'ves_seg', f"{base_tomo_name}_wbp_corrected.mrc")
                label_path = os.path.join(current_path, tomo_name, 'ves_seg', f"{base_tomo_name}_label_vesicle.mrc")

                # 检查文件是否存在
                if not all(os.path.exists(p) for p in [deconv_tomo_path, isonet_tomo_path, label_path]):
                    self.print_callback(f"部分关键文件不存在，跳过 tomo: {tomo_name}")
                    continue

                tomo_paths.append({
                    'tomo_name': tomo_name,
                    'base_tomo_name': base_tomo_name,
                    'deconv_tomo_path': deconv_tomo_path,
                    'isonet_tomo_path': isonet_tomo_path,
                    'label_path': label_path
                })

            if not tomo_paths:
                self.print_callback("没有有效的 tomo 需要处理。")
                return

            # 创建 subtomo 文件夹及其子文件夹
            subtomo_dir = os.path.join(current_path, 'subtomo')
            iso_dir = os.path.join(subtomo_dir, 'iso')
            dec_dir = os.path.join(subtomo_dir, 'dec')
            label_dir = os.path.join(subtomo_dir, 'label')

            os.makedirs(iso_dir, exist_ok=True)
            os.makedirs(dec_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)

            self.print_callback(f"创建切片保存目录: {subtomo_dir}")

            image_list = []
            label_list = []
            slice_counter = 0

            # 对每个 tomo 的三个关键文件进行切片处理
            for tomo in tomo_paths:
                self.print_callback(f"处理 tomo: {tomo['tomo_name']}")

                # 读取 deconv 文件并转换为 int8
                with mrcfile.open(tomo['deconv_tomo_path']) as f:
                    dec_data = f.data
                dec_data = np.pad(dec_data, 40, 'constant', constant_values=np.mean(dec_data)).astype(np.int8)

                # 读取 isonet 文件并转换为 int8
                with mrcfile.open(tomo['isonet_tomo_path']) as f:
                    iso_data = f.data
                iso_data = np.pad(iso_data, 40, 'constant', constant_values=np.mean(iso_data)).astype(np.int8)

                # 读取 label 文件并转换为 int16
                with mrcfile.open(tomo['label_path']) as f:
                    label_data = f.data
                label_data = np.pad(label_data, 40, 'constant', constant_values=0).astype(np.int16)

                # 计算连通体
                labeled = label(label_data)
                regions = regionprops(labeled)
                self.print_callback(f"找到 {len(regions)} 个连通体。")

                for region in regions:
                    center = np.round(region.centroid).astype(np.int16)
                    x, y, z = center

                    # 确保切片不会超出数据边界
                    sbtm_iso = iso_data[x-32:x+32, y-32:y+32, z-32:z+32]
                    sbtm_dec = dec_data[x-32:x+32, y-32:y+32, z-32:z+32]
                    sblbl = label_data[x-32:x+32, y-32:y+32, z-32:z+32]
                    sblbl = np.sign(sblbl).astype(np.int8)

                    # 切片命名
                    slice_name = f"vesicle_{str(slice_counter).zfill(8)}"

                    # 保存 iso 切片
                    iso_slice_path = os.path.join(iso_dir, f"{slice_name}.mrc")
                    with mrcfile.new(iso_slice_path, overwrite=True) as m:
                        m.set_data(sbtm_iso)
                    image_list.append(os.path.basename(iso_slice_path))

                    # 保存 dec 切片
                    dec_slice_path = os.path.join(dec_dir, f"{slice_name}.mrc")
                    with mrcfile.new(dec_slice_path, overwrite=True) as m:
                        m.set_data(sbtm_dec)

                    # 保存 label 切片
                    label_slice_path = os.path.join(label_dir, f"{slice_name}_label.mrc")
                    with mrcfile.new(label_slice_path, overwrite=True) as m:
                        m.set_data(sblbl)
                    label_list.append(os.path.basename(label_slice_path))

                    slice_counter += 1

            self.print_callback(f"总共切片数量: {slice_counter}")

            # 保存名称列表
            image_txt_path = os.path.join(subtomo_dir, 'image.txt')
            label_txt_path = os.path.join(subtomo_dir, 'label.txt')

            with open(image_txt_path, 'w') as f:
                for img in image_list:
                    f.write(f"{img}\n")
            self.print_callback(f"保存 image.txt 到: {image_txt_path}")

            with open(label_txt_path, 'w') as f:
                for lbl in label_list:
                    f.write(f"{lbl}\n")
            self.print_callback(f"保存 label.txt 到: {label_txt_path}")

            # 执行训练函数
            dim_in = 64
            datadir = subtomo_dir

            # 第一次训练
            self.print_callback("开始第一次训练 DEC 模型...")
            train_model(dim_in, batch_size, epochs, dec_model_pth, datadir, mode='dec')
            self.print_callback("第一次训练 DEC 模型完成。")

            # 第二次训练
            self.print_callback("开始第二次训练 ISO 模型...")
            train_model(dim_in, batch_size, epochs, iso_model_pth, datadir, mode='iso')
            self.print_callback("第二次训练 ISO 模型完成。")

            self.print_callback("finetune_model 函数执行完毕。")

        except Exception as e:
            self.print_callback(f"执行 finetune_model 时发生错误: {e}")
