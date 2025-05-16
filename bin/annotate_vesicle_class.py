import os
import xml.etree.ElementTree as ET
import numpy as np
import mrcfile

from qtpy.QtWidgets import (
    QMainWindow, QWidget, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QComboBox, QApplication
)
from qtpy.QtCore import Qt
import napari

LABEL_OPTIONS = ['false', 'tether', 'contact', 'omega', 'pit', 'CCV', 'endosome', 'DCV', 'others']


import os
import json
import re

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QListWidgetItem,
    QCheckBox, QPushButton, QFileDialog, QHBoxLayout, QLabel, QSizePolicy
)
from qtpy.QtGui import QPixmap, QIcon
from qtpy.QtCore import Qt

class FolderListWidget(QWidget):
    """
    文件夹列表控件，用于选择 tomogram 文件夹并触发主窗口加载。
    """
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        # 当前根路径，由主窗口提供
        self.current_path = parent.current_path

        # 默认状态文件
        self.state_file = os.path.join(self.current_path, 'segVesicle_QCheckBox_state.json')
        self.heart_state_file = os.path.join(self.current_path, 'segVesicle_heart_broken.json')
        self.checkbox_states = self._load_json(self.state_file)
        self.heart_states = self._load_json(self.heart_state_file)

        # UI 组件
        self.layout = QVBoxLayout(self)
        self.open_folder_btn = QPushButton("Open Folder")
        self.open_folder_btn.clicked.connect(self.open_folder_dialog)
        self.layout.addWidget(self.open_folder_btn)

        self.create_batch_btn = QPushButton("Create segVesicle.batch")
        self.create_batch_btn.clicked.connect(self.create_segvesicle_batch)
        self.layout.addWidget(self.create_batch_btn)

        self.list_widget = QListWidget()
        self.layout.addWidget(self.list_widget)

        # 初始填充
        self.populate_list()
        # 绑定双击
        self.list_widget.itemDoubleClicked.connect(self._on_double_click)

    def _load_json(self, path):
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_json(self, path, data):
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

    def open_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Root Folder", self.current_path)
        if folder:
            self.current_path = folder
            self.parent.current_path = folder
            # 更新状态文件路径
            self.state_file = os.path.join(folder, 'segVesicle_QCheckBox_state.json')
            self.heart_state_file = os.path.join(folder, 'segVesicle_heart_broken.json')
            self.checkbox_states = self._load_json(self.state_file)
            self.heart_states = self._load_json(self.heart_state_file)
            self.populate_list()

    def create_segvesicle_batch(self):
        current = self.current_path
        top_folders = set()
        for root, dirs, files in os.walk(current):
            if root == current:
                continue
            if any(f.endswith(('.mrc', '.rec')) for f in files):
                rel = os.path.relpath(root, current).split(os.sep)[0]
                top_folders.add(rel)
                dirs[:] = []
        with open(os.path.join(current, 'segVesicle.batch'), 'w') as f:
            for d in sorted(top_folders):
                f.write(d + '\n')
        self.populate_list()

    def populate_list(self):
        self.list_widget.clear()
        batch_file = os.path.join(self.current_path, 'segVesicle.batch')
        if not os.path.exists(batch_file):
            return
        with open(batch_file) as f:
            lines = [l.strip() for l in f if l.strip()]
        def sort_key(x):
            m = re.match(r"(pp?)(\d+)", x)
            return (m.group(1), int(m.group(2))) if m else (x, 0)
        folders = sorted(set(lines), key=sort_key)

        for name in folders:
            item = QListWidgetItem()
            widget = QWidget()
            h = QHBoxLayout(widget)
            h.setContentsMargins(2, 2, 2, 2)
            # 心形复选框
            heart = QCheckBox()
            heart_icon = QIcon(QPixmap('resource/imgs/heart.svg'))
            broken_icon = QIcon(QPixmap('resource/imgs/broken_heart.svg'))
            heart.setIcon(heart_icon)
            if self.heart_states.get(name, False):
                heart.setChecked(True)
                heart.setIcon(broken_icon)
            heart.stateChanged.connect(lambda s, n=name, cb=heart: self._toggle_heart(s, n, cb, heart_icon, broken_icon))
            h.addWidget(heart)
            # 普通复选框
            cb = QCheckBox()
            cb.setChecked(self.checkbox_states.get(name, False))
            cb.stateChanged.connect(lambda s, n=name: self._toggle_check(s, n))
            h.addWidget(cb)
            # 标签
            lbl = QLabel(name)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            h.addWidget(lbl)
            h.addStretch()

            item.setSizeHint(widget.sizeHint())
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, widget)

    def _toggle_heart(self, state, name, checkbox, icon, broken_icon):
        checked = (state == Qt.Checked)
        self.heart_states[name] = checked
        checkbox.setIcon(broken_icon if checked else icon)
        self._save_json(self.heart_state_file, self.heart_states)

    def _toggle_check(self, state, name):
        self.checkbox_states[name] = (state == Qt.Checked)
        self._save_json(self.state_file, self.checkbox_states)

    def _on_double_click(self, item):
        # 接收 QListWidgetItem 或直接字符串
        if isinstance(item, QListWidgetItem):
            widget = self.list_widget.itemWidget(item)
            label = widget.findChild(QLabel)
            name = label.text() if label else None
        elif isinstance(item, str):
            name = item
        else:
            name = None
        if name:
            self.parent.on_item_double_click(name)


def get_tomo(path):
    """
    Load a 3D MRC file as a numpy array.
    """
    with mrcfile.open(path) as mrc:
        return mrc.data


class AnnotateVesicleWindow(QMainWindow):
    def __init__(self, folder_path):
        super().__init__()
        self.current_path = folder_path
        self.base_tomo_name = None
        self.filter_xml_path = None
        self.class_xml_path = None
        self.current_index = 0

        # XML 数据
        self.tree = None
        self.root = None
        self.vesicles = []
        self.classes = {}

        # Napari 子窗口
        self.patch_viewers = [napari.Viewer(show=False) for _ in range(12)]
        self.patch_layers = [None] * 11
        self.class_inputs = [None] * 11

        # 构建 UI
        self.init_ui()
        self.setWindowTitle('Annotate Vesicle Tool')
        self.show()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # 左侧：标注网格
        grid_widget = QWidget()
        grid_layout = QGridLayout()
        for i in range(12):
            row, col = divmod(i, 4)
            if i == 11:
                btn_prev = QPushButton('Previous')
                btn_prev.clicked.connect(self.prev_vesicles)
                btn_next = QPushButton('Next')
                btn_next.clicked.connect(self.next_vesicles)
                nav_layout = QHBoxLayout()
                nav_layout.addWidget(btn_prev)
                nav_layout.addWidget(btn_next)
                grid_layout.addLayout(nav_layout, row, col)
            else:
                v_layout = QVBoxLayout()
                v_layout.addWidget(self.patch_viewers[i].window.qt_viewer)
                combo = QComboBox()
                combo.addItems(LABEL_OPTIONS)
                combo.currentTextChanged.connect(lambda val, idx=i: self.update_class(idx, val))
                v_layout.addWidget(combo)
                self.class_inputs[i] = combo
                grid_layout.addLayout(v_layout, row, col)
        grid_widget.setLayout(grid_layout)
        main_layout.addWidget(grid_widget, stretch=3)

        # 右侧：文件夹列表
        self.folder_list = FolderListWidget(self)
        main_layout.addWidget(self.folder_list, stretch=1)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def on_item_double_click(self, tomo_name: str):
        # 选定 tomo 文件夹名，接收字符串
        self.base_tomo_name = tomo_name.split('-1')[0] if '-1' in tomo_name else tomo_name
        # 构建 XML 路径
        self.filter_xml_path = os.path.join(
            self.current_path, tomo_name, 'ves_seg', 'vesicle_analysis',
            f'{self.base_tomo_name}_filter.xml'
        )
        self.class_xml_path = os.path.join(
            self.current_path, tomo_name, 'ves_seg', 'vesicle_analysis',
            f'{self.base_tomo_name}_vesicle_class.xml'
        )
        # 读取 XML
        self.load_vesicles_from_xml()
        self.current_index = 0
        self.display_current_vesicles()

    def load_vesicles_from_xml(self):
        # 优先使用 class_xml，否则 fallback 到 filter_xml
        xml_path = self.class_xml_path if os.path.exists(self.class_xml_path) else self.filter_xml_path
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # 查找所有 Vesicle 节点并过滤 Type t != 'vesicle'
        all_ves = root.findall('Vesicle')
        ves_list = []
        for v in all_ves:
            t_elem = v.find('Type')
            if t_elem is not None and t_elem.attrib.get('t') != 'vesicle' and t_elem.attrib.get('t') != 'pit':
                ves_list.append(v)
        # 初始化类别字典
        classes = {}
        for v in ves_list:
            vid = v.attrib.get('vesicleId')
            t_elem = v.find('Type')
            classes[vid] = t_elem.attrib.get('t') if t_elem is not None else 'others'
        self.tree = tree
        self.root = root
        self.vesicles = ves_list
        self.classes = classes

    def display_current_vesicles(self):
        total = len(self.vesicles)
        for i in range(11):
            idx = self.current_index + i
            combo = self.class_inputs[i]
            viewer = self.patch_viewers[i]
            if self.patch_layers[i] is not None:
                viewer.layers.remove(self.patch_layers[i])
                self.patch_layers[i] = None

            if idx < total:
                vid = self.vesicles[idx].attrib.get('vesicleId')
                mrc_path = os.path.join(
                    self.current_path, self.base_tomo_name, 'ves_seg', 'vesicle_analysis',
                    'extractRRP_3D', f'vesicle_{vid}.mrc'
                )
                if os.path.exists(mrc_path):
                    data = get_tomo(mrc_path)
                    vmin, vmax = np.percentile(data, [0.1, 99])
                    layer = viewer.add_image(data, contrast_limits=[vmin, vmax], name=f'#{idx}')
                    self.patch_layers[i] = layer
                vid = self.vesicles[idx].attrib.get('vesicleId')
                combo.setEnabled(True)
                combo.setCurrentText(self.classes.get(vid, 'others'))
            else:
                combo.setEnabled(False)
                combo.setCurrentText('others')

    def update_class(self, idx, value):
        ves_idx = self.current_index + idx
        if ves_idx < len(self.vesicles):
            vid = self.vesicles[ves_idx].attrib.get('vesicleId')
            self.classes[vid] = value

    def prev_vesicles(self):
        self.save_current_classes()
        if self.current_index >= 11:
            self.current_index -= 11
            self.display_current_vesicles()

    def next_vesicles(self):
        self.save_current_classes()
        if self.current_index + 11 < len(self.vesicles):
            self.current_index += 11
            self.display_current_vesicles()

    def save_current_classes(self):
        for i in range(11):
            idx = self.current_index + i
            if idx < len(self.vesicles):
                ves = self.vesicles[idx]
                # 1) 移除所有旧的 <Type>
                for old in ves.findall('Type'):
                    ves.remove(old)

                # 2) 创建并设置新的 <Type>
                vid = ves.attrib.get('vesicleId')
                new_t = ET.SubElement(ves, 'Type')
                new_t.set('t', self.classes.get(vid, 'others'))

        # 3) 写回 XML
        self.tree.write(self.class_xml_path, encoding='utf-8', xml_declaration=False)


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    folder = os.getcwd()
    win = AnnotateVesicleWindow(folder)
    sys.exit(app.exec_())
