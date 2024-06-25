import sys
import os
from packaging.version import parse as parse_version
from copy import deepcopy
import numpy as np
from datetime import datetime
from qtpy.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QWidget,
    QPushButton,
    QApplication, 
    QVBoxLayout, 
    QMainWindow,
    QTextBrowser,
    QTextEdit
)
from qtpy.QtCore import QProcess, QByteArray, Qt, QEvent, Signal, QObject
from qtpy import uic, QtGui, QtCore
from superqt.utils import qthrottled
import napari
from napari.components.layerlist import Extent
from napari.components.viewer_model import ViewerModel
from napari.layers import Image, Labels, Vectors, Layer
from napari.qt import QtViewer
from napari.utils.notifications import show_info
from napari.utils.events.event import WarningEmitter
from napari.utils.action_manager import action_manager
from resource.Ui_utils_widge import Ui_Form
from global_vars import global_viewer

# 判断当前 napari 版本是否大于 0.4.16
NAPARI_GE_4_16 = parse_version(napari.__version__) > parse_version("0.4.16")
      
class UtilWidge(QWidget):
    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.update_progress_stage()
        self.ui.help.clicked.connect(self.show_help)
        
    def update_progress_stage(self):
        # 将全局变量的内容显示在 QTextEdit 中
        from launch import TOMO_SEGMENTATION_PROGRESS
        self.ui.progressStage.setText(str(TOMO_SEGMENTATION_PROGRESS))
        
    @QtCore.Slot(str)  # 标记这个方法是一个槽
    def print_in_widget(self, text):
        # self.ui.terminal.append(text)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.ui.terminal.append(f"[{current_time}] {text}")
        
    def show_help(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        markdown_file = os.path.join(script_dir, 'resource', 'help.md')

        # 创建一个新的 QMainWindow
        self.help_window = QMainWindow(self.viewer.window.qt_viewer)
        self.help_window.setWindowTitle("Help Documentation")
        self.help_window.setGeometry(100, 100, 600, 400)

        # 创建一个空的 QWidget
        central_widget = QWidget(self.help_window)
        self.help_window.setCentralWidget(central_widget)

        # 创建布局
        layout = QVBoxLayout(central_widget)

        # 创建 QTextBrowser 显示 .md 文件内容
        text_browser = QTextBrowser(self.help_window)
        
        # 读取 .md 文件内容
        with open(markdown_file, 'r') as file:
            markdown_content = file.read()
        
        # 设置 .md 文件内容到 QTextBrowser
        text_browser.setMarkdown(markdown_content)

        # 添加 QTextBrowser 到布局
        layout.addWidget(text_browser)

        # 显示窗口
        self.help_window.show()
        self.help_window.raise_()  # 确保窗口出现在最前面


def copy_layer_le_4_16(layer: Layer, name: str = ""):
    """
    复制一个 Layer 对象（适用于 napari 版本 > 0.4.16）。
    
    参数:
    layer (Layer): 要复制的图层
    name (str): 图层的新名称

    返回:
    Layer: 复制后的图层
    """
    res_layer = deepcopy(layer)
    if isinstance(layer, (Image, Labels)):
        res_layer.data = layer.data
    res_layer.metadata["viewer_name"] = name
    res_layer.events.disconnect()
    res_layer.events.source = res_layer
    for emitter in res_layer.events.emitters.values():
        emitter.disconnect()
        emitter.source = res_layer
    return res_layer


def copy_layer(layer: Layer, name: str = ""):
    """
    根据 napari 版本选择合适的图层复制方法。
    """
    if NAPARI_GE_4_16:
        return copy_layer_le_4_16(layer, name)
    res_layer = Layer.create(*layer.as_layer_data_tuple())
    res_layer.metadata["viewer_name"] = name
    return res_layer


def get_property_names(layer: Layer):
    """
    获取一个图层对象的所有可设置属性名称。

    参数:
    layer (Layer): 一个 napari 图层对象。

    返回:
    List[str]: 图层对象的所有可设置属性名称列表。
    """
    klass = layer.__class__
    res = []
    for event_name, event_emitter in layer.events.emitters.items():
        # 跳过警告
        if isinstance(event_emitter, WarningEmitter):
            continue
        # 跳过不需要的属性
        if event_name in ("thumbnail", "name"):
            continue
        # 检查属性是否为property并且具有setter方法
        if (
            isinstance(getattr(klass, event_name, None), property)
            and getattr(klass, event_name).fset is not None
        ):
            res.append(event_name)
    return res


class CrossWidget(QCheckBox):
    """
    CrossWidget 类用于在 napari 视图中添加一个可见的cross layer。
    该类继承自 QCheckBox, 并根据用户交互和视图变化动态更新交叉层。
    """
    def __init__(self, viewer: napari.Viewer) -> None:
        # 调用父类 QCheckBox 的构造方法，并设置复选框的标签为 "Add cross layer"。
        super().__init__("Add cross layer")
        self.viewer = viewer
        # 初始状态下复选框未选中。
        self.setChecked(False)
        # 当复选框状态改变时，调用 _update_cross_visibility 方法。
        self.stateChanged.connect(self._update_cross_visibility)
        self.layer = None
        # 视图维度顺序改变时，调用 update_cross 方法。
        self.viewer.dims.events.order.connect(self.update_cross)
        # 视图维度数目改变时，调用 _update_ndim 方法。
        self.viewer.dims.events.ndim.connect(self._update_ndim)
        # 视图当前步数改变时，调用 update_cross 方法。
        self.viewer.dims.events.current_step.connect(self.update_cross)
        self._extent = None
        self._update_extent()
        # 视图维度事件发生时，调用 _update_extent 方法。
        self.viewer.dims.events.connect(self._update_extent)

    @qthrottled(leading=False) # 装饰器，限制方法的调用频率，防止频繁调用导致性能问题。
    def _update_extent(self):
        '''
        更新视图的范围信息
        '''
        if NAPARI_GE_4_16:
            layers = [layer for layer in self.viewer.layers if layer is not self.layer]
            self._extent = self.viewer.layers.get_extent(layers)
        else:
            extent_list = [layer.extent for layer in self.viewer.layers if layer is not self.layer]
            self._extent = Extent(
                data=None,
                world=self.viewer.layers._get_extent_world(extent_list),
                step=self.viewer.layers._get_step_size(extent_list),
            )
        self.update_cross()

    def _update_ndim(self, event):
        '''
        更新维度信息，当维度变化时，重新创建交叉层。
        '''
        if self.layer in self.viewer.layers:
            self.viewer.layers.remove(self.layer)
        self.layer = Vectors(name=".cross", ndim=event.value, vector_style='line', edge_color='yellow')
        self.layer.edge_width = 1.5
        self.update_cross()

    def _update_cross_visibility(self, state):
        """
        根据复选框状态显示或隐藏交叉层。
        """
        if state:
            self.viewer.layers.append(self.layer)
            self.viewer.layers.move(self.viewer.layers.index(self.layer), 0)
        else:
            self.viewer.layers.remove(self.layer)
        self.update_cross()

    def update_cross(self):
        """
        更新交叉层的数据和显示。
        """
        if self.layer not in self.viewer.layers:
            return
        point = self.viewer.dims.current_step
        vec = []
        for i, (lower, upper) in enumerate(self._extent.world.T):
            if (upper - lower) / self._extent.step[i] == 1:
                continue
            point1 = list(point)
            point1[i] = (lower + self._extent.step[i] / 2) / self._extent.step[i]
            point2 = [0 for _ in point]
            point2[i] = (upper - lower) / self._extent.step[i]
            vec.append((point1, point2))
        if np.any(self.layer.scale != self._extent.step):
            self.layer.scale = self._extent.step
        self.layer.data = vec


class MultipleViewerWidget(QWidget):
    message_signal = QtCore.Signal(str)  # 定义一个信号
    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self.viewer_model1 = ViewerModel(title="xz_ortho")
        self.viewer_model2 = ViewerModel(title="yz_ortho")
        self.viewer_model3 = ViewerModel(title="xy_ortho")
        self.viewer_model3.dims.ndisplay = 2 + (viewer.dims.ndisplay == 2)
        self._block = False
        self.qt_viewer1 = QtViewerWrap(viewer, self.viewer_model1)
        self.qt_viewer2 = QtViewerWrap(viewer, self.viewer_model2)
        self.qt_viewer3 = viewer.window.qt_viewer
        
        # self.utils_widget = QWidget()
        self.utils_widget = UtilWidge(self.viewer)
        # uic.loadUi('resource/utils_widge.ui', self.utils_widget)
        
        grid_layout = QGridLayout()
        grid_layout.addWidget(self.utils_widget, 0, 1)
        grid_layout.addWidget(self.qt_viewer1, 0, 0)
        grid_layout.addWidget(self.qt_viewer2, 1, 1)
        grid_layout.addWidget(self.qt_viewer3, 1, 0)
        grid_layout.setRowStretch(0, 1)
        grid_layout.setRowStretch(1, 3)
        grid_layout.setColumnStretch(0, 3)
        grid_layout.setColumnStretch(1, 1)
        self.setLayout(grid_layout)
        self.viewer.layers.events.inserted.connect(self._layer_added)
        self.viewer.layers.events.removed.connect(self._layer_removed)
        self.viewer.layers.events.moved.connect(self._layer_moved)
        self.viewer.layers.selection.events.active.connect(self._layer_selection_changed)
        self.viewer.dims.events.current_step.connect(self._point_update)
        self.viewer_model1.dims.events.current_step.connect(self._point_update)
        self.viewer_model2.dims.events.current_step.connect(self._point_update)
        self.viewer_model3.dims.events.current_step.connect(self._point_update)
        self.viewer.dims.events.order.connect(self._order_update)
        self.viewer.events.reset_view.connect(self._reset_view)
        self.viewer_model1.events.status.connect(self._status_update)
        self.viewer_model2.events.status.connect(self._status_update)
        self.viewer_model3.events.status.connect(self._status_update)
        
        # 连接信号到槽
        self.message_signal.connect(self.utils_widget.print_in_widget)
        
    def print_in_widget(self, text):
        self.utils_widget.print_in_widget(text)
        

    def show_deconvolution_widget(self):
        # # 创建空的 QWidget
        # self.deconvolution_widget = QWidget()
        
        # # 加载 UI 文件
        # uic.loadUi('/Users/shor/Documents/napari_demo/resourse/deconvolution.ui', self.deconvolution_widget)

        # # 创建一个新的窗口来显示 deconvolution_widget
        # self.deconvolution_window = QMainWindow()
        # self.deconvolution_window.setCentralWidget(self.deconvolution_widget)
        # self.deconvolution_window.setWindowTitle("Deconvolution Widget")
        
        # # 显示窗口
        # self.deconvolution_window.show()
        
        # 创建一个新的 QMainWindow
        self.deconvolution_window = QMainWindow(self.viewer.window.qt_viewer)
        
        # 创建一个空的 QWidget
        self.deconvolution_widget = QWidget(self.deconvolution_window)
        
        # 加载 UI 文件
        uic.loadUi('/Users/shor/Documents/napari_demo/resourse/deconvolution.ui', self.deconvolution_widget)
        
        # 设置 deconvolution_widget 为中央部件
        self.deconvolution_window.setCentralWidget(self.deconvolution_widget)
        self.deconvolution_window.setWindowTitle("Deconvolution Widget")
        
        # 显示窗口
        self.deconvolution_window.show()

    def _status_update(self, event):
        """
        更新状态: 将事件的状态值赋给主 viewer 的状态。
        """
        self.viewer.status = event.value

    def _reset_view(self):
        '''
        重置视图: 重置四个 viewer 模型的视图。
        '''
        self.viewer_model1.reset_view()
        self.viewer_model2.reset_view()
        self.viewer_model3.reset_view()

    def _layer_selection_changed(self, event):
        '''
        更新选中的图层: 根据事件更新3个 viewer 模型中选中的图层。
        '''
        if self._block:
            return
        if event.value is None:
            self.viewer_model1.layers.selection.active = None
            self.viewer_model2.layers.selection.active = None
            self.viewer_model3.layers.selection.active = None
            return
        self.viewer_model1.layers.selection.active = self.viewer_model1.layers[event.value.name]
        self.viewer_model2.layers.selection.active = self.viewer_model2.layers[event.value.name]
        self.viewer_model3.layers.selection.active = self.viewer_model3.layers[event.value.name]

    def _point_update(self, event):
        '''
        更新当前步数: 将事件的当前步数值赋给3个 viewer 模型。
        '''
        for model in [self.viewer, self.viewer_model1, self.viewer_model2, self.viewer_model3]:
            if model.dims is event.source:
                continue
            if len(self.viewer.layers) != len(model.layers):
                continue
            model.dims.current_step = event.value

    def _order_update(self):
        '''
        更新维度顺序: 根据主 viewer 的维度顺序更新3个 viewer 模型的维度顺序。
        '''
        order = list(self.viewer.dims.order)
        if len(order) <= 2:
            self.viewer_model1.dims.order = order
            self.viewer_model2.dims.order = order
            self.viewer_model3.dims.order = order
            return
        order[-3:] = order[-2], order[-3], order[-1]
        self.viewer_model1.dims.order = order
        order = list(self.viewer.dims.order)
        order[-3:] = order[-1], order[-2], order[-3]
        self.viewer_model2.dims.order = order

    def _layer_added(self, event):
        '''
        添加图层: 将新图层添加到3个 viewer 模型中，并连接相应的事件处理方法。
        '''
        self.viewer_model1.layers.insert(event.index, copy_layer(event.value, "model1"))
        self.viewer_model2.layers.insert(event.index, copy_layer(event.value, "model2"))
        self.viewer_model3.layers.insert(event.index, copy_layer(event.value, "model3"))
        for name in get_property_names(event.value):
            getattr(event.value.events, name).connect(own_partial(self._property_sync, name))
        if isinstance(event.value, Labels):
            event.value.events.set_data.connect(self._set_data_refresh)
            self.viewer_model1.layers[event.value.name].events.set_data.connect(self._set_data_refresh)
            self.viewer_model2.layers[event.value.name].events.set_data.connect(self._set_data_refresh)
            self.viewer_model3.layers[event.value.name].events.set_data.connect(self._set_data_refresh)
        if event.value.name != ".cross":
            self.viewer_model1.layers[event.value.name].events.data.connect(self._sync_data)
            self.viewer_model2.layers[event.value.name].events.data.connect(self._sync_data)
            self.viewer_model3.layers[event.value.name].events.data.connect(self._sync_data)
        event.value.events.name.connect(self._sync_name)
        self._order_update()

    def _sync_name(self, event):
        '''
        同步图层名称: 同步3个 viewer 模型中图层的名称。
        '''
        index = self.viewer.layers.index(event.source)
        self.viewer_model1.layers[index].name = event.source.name
        self.viewer_model2.layers[index].name = event.source.name
        self.viewer_model3.layers[index].name = event.source.name

    def _sync_data(self, event):
        '''
        同步图层数据: 同步3个 viewer 模型中图层的数据。
        '''
        if self._block:
            return
        for model in [self.viewer, self.viewer_model1, self.viewer_model2, self.viewer_model3]:
            layer = model.layers[event.source.name]
            if layer is event.source:
                continue
            try:
                self._block = True
                layer.data = event.source.data
            finally:
                self._block = False

    def _set_data_refresh(self, event):
        '''
        刷新图层数据: 刷新3个 viewer 模型中图层的数据。
        '''
        if self._block:
            return
        for model in [self.viewer, self.viewer_model1, self.viewer_model2, self.viewer_model3]:
            layer = model.layers[event.source.name]
            if layer is event.source:
                continue
            try:
                self._block = True
                layer.refresh()
            finally:
                self._block = False

    def _layer_removed(self, event):
        self.viewer_model1.layers.pop(event.index)
        self.viewer_model2.layers.pop(event.index)
        self.viewer_model3.layers.pop(event.index)

    def _layer_moved(self, event):
        dest_index = event.new_index if event.new_index < event.index else event.new_index + 1
        self.viewer_model1.layers.move(event.index, dest_index)
        self.viewer_model2.layers.move(event.index, dest_index)
        self.viewer_model3.layers.move(event.index, dest_index)

    def _property_sync(self, name, event):
        if event.source not in self.viewer.layers:
            return
        try:
            self._block = True
            setattr(self.viewer_model1.layers[event.source.name], name, getattr(event.source, name))
            setattr(self.viewer_model2.layers[event.source.name], name, getattr(event.source, name))
            setattr(self.viewer_model3.layers[event.source.name], name, getattr(event.source, name))
        finally:
            self._block = False


class own_partial:
    def __init__(self, func, *args, **kwargs) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.func(*(self.args + args), **{**self.kwargs, **kwargs})

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        return own_partial(self.func, *deepcopy(self.args, memodict), **deepcopy(self.kwargs, memodict))


class QtViewerWrap(QtViewer):
    def __init__(self, main_viewer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.main_viewer = main_viewer
        self.setAcceptDrops(True)

    def _qt_open(self, filenames: list, stack: bool, plugin: str = None, layer_type: str = None, **kwargs):
        self.main_viewer.window._qt_viewer._qt_open(filenames, stack, plugin, layer_type, **kwargs)

    def dropEvent(self, event):
        self.main_viewer.window._qt_viewer.dropEvent(event)
