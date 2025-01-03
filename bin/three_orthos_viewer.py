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
)
from qtpy.QtCore import QTimer, QProcess, QByteArray, Qt, QEvent, Signal, QObject, QThread
from qtpy import uic, QtGui, QtCore
import napari
from napari.components.layerlist import Extent
from napari.components.viewer_model import ViewerModel
from napari.layers import Image, Labels, Vectors, Layer
from napari.qt import QtViewer
from napari.utils.notifications import show_info
from napari.utils.events.event import WarningEmitter
from napari.utils.action_manager import action_manager
from resource.Ui_utils_widge import Ui_Form
from global_vars import TOMO_NAME
from help_viewer import HelpViewer
import threading

# 判断当前 napari 版本是否大于 0.4.16
NAPARI_GE_4_16 = parse_version(napari.__version__) > parse_version("0.4.16") and parse_version(napari.__version__) < parse_version("0.5.0")
# NAPARI_GE_4_16 = parse_version(napari.__version__) > parse_version("0.4.16")

import concurrent.futures
from qtpy.QtCore import QTimer, Signal, QObject
from qtpy.QtWidgets import QWidget, QGridLayout, QMainWindow
import napari

class Worker(QObject):
    finished = Signal()
    progress = Signal(int)

    def __init__(self, viewer, viewer_model1, viewer_model2, viewer_model3):
        super().__init__()
        self.viewer = viewer
        self.viewer_model1 = viewer_model1
        self.viewer_model2 = viewer_model2
        self.viewer_model3 = viewer_model3
        self._block = False

    def run(self, event):
        self._block = True
        try:
            current_steps = [self.viewer.dims.current_step,
                             self.viewer_model1.dims.current_step,
                             self.viewer_model2.dims.current_step,
                             self.viewer_model3.dims.current_step]
            
            if all(step == self.viewer.dims.current_step for step in current_steps):
                return

            for model in [self.viewer_model1, self.viewer_model2, self.viewer_model3]:
                if model.dims is not event.source and len(self.viewer.layers) == len(model.layers):
                    model.dims.current_step = self.viewer.dims.current_step
        finally:
            self._block = False
        self.finished.emit()

class WorkerForCross(QObject):
    finished = Signal()
    extent_updated = Signal(object)

    def __init__(self, viewer, layer):
        super().__init__()
        self.viewer = viewer
        self.layer = layer

    def run(self):
        if NAPARI_GE_4_16:
            layers = [layer for layer in self.viewer.layers if layer is not self.layer]
            extent = self.viewer.layers.get_extent(layers)
        else:
            extent_list = [layer.extent for layer in self.viewer.layers if layer is not self.layer]
            extent = Extent(
                data=None,
                world=self.viewer.layers._get_extent_world(extent_list),
                step=self.viewer.layers._get_step_size(extent_list),
            )
        self.extent_updated.emit(extent)
        self.finished.emit()

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
        # from launch import TOMO_SEGMENTATION_PROGRESS
        # self.ui.progressStage.setText(str(TOMO_SEGMENTATION_PROGRESS))
        pass
        
    def print_in_widget(self, text):
        from qtpy.QtCore import QMetaObject, Q_ARG
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_text = f"[{current_time}] {text}"
        QMetaObject.invokeMethod(self.ui.terminal, "append", Qt.QueuedConnection, Q_ARG(str, formatted_text))
        
    def show_help(self):
        self.help_window = HelpViewer()
        # 显示窗口
        self.help_window.show()
        self.help_window.raise_()  # 确保窗口出现在最前面


def copy_layer_le_4_16(layer: Layer, name: str = ''):
    res_layer = deepcopy(layer)
    # this deepcopy is not optimal for labels and images layers
    if isinstance(layer, (Image, Labels)):
        res_layer.data = layer.data
    res_layer.metadata['viewer_name'] = name

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
        self._extent = None
        self.pending_update = False
        
        self.update_timer = QTimer()
        self.update_timer.setInterval(50)  # 100毫秒更新一次
        self.update_timer.timeout.connect(self._process_pending_updates)
        
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        
        # 视图维度顺序改变时，调用 update_cross 方法。
        self.viewer.dims.events.order.connect(self.update_cross)
        # self.viewer.dims.events.order.connect(self._schedule_update_extent)

        # 视图维度数目改变时，调用 _update_ndim 方法。
        self.viewer.dims.events.ndim.connect(self._update_ndim)
        # 视图当前步数改变时，调用 update_cross 方法。
        # self.viewer.dims.events.current_step.connect(self.update_cross)
        self.viewer.dims.events.current_step.connect(self.update_cross)
        # 视图维度事件发生时，调用 _update_extent 方法。
        # self.viewer.dims.events.connect(self._update_extent)
        self.viewer.dims.events.connect(self._schedule_update_extent)
        
        # self._update_extent()


    def _schedule_update_extent(self, event=None):
        self.pending_update = True
        if not self.update_timer.isActive():
            self.update_timer.start()

    def _process_pending_updates(self):
        if self.pending_update:
            self.pending_update = False
            self._run_update_extent()

    def _run_update_extent(self):
        worker = WorkerForCross(self.viewer, self.layer)
        worker.extent_updated.connect(self._on_extent_updated)
        worker.finished.connect(self._on_worker_finished)
        self.executor.submit(worker.run)

    def _on_extent_updated(self, extent):
        self._extent = extent

    def _on_worker_finished(self):
        self.update_cross()

    def _update_ndim(self, event):
        '''
        更新维度信息，当维度变化时，重新创建交叉层。
        '''
        if self.layer in self.viewer.layers:
            self.viewer.layers.remove(self.layer)
        # self.layer = Vectors(name=".cross", ndim=event.value, vector_style='line', edge_color='yellow')
        # self.layer.edge_width = 1.5
        # 创建 Vectors 图层
        self.layer = Vectors(name=".cross", ndim=event.value, vector_style='line', edge_color='yellow')

        # 设置图层属性
        self.layer.edge_width = 2.5
        self.layer.opacity = 1
        self.update_cross()

    def _update_cross_visibility(self, state):
        """
        根据复选框状态显示或隐藏交叉层。
        """
        if state:
            self.viewer.layers.append(self.layer)
            self.viewer.layers.move(self.viewer.layers.index(self.layer), -1)
            self.viewer.layers.selection.active = self.viewer.layers['edit vesicles']
        else:
            self.viewer.layers.remove(self.layer)
            self.viewer.layers.selection.active = self.viewer.layers['edit vesicles']
        self.update_cross()

    def update_cross(self):
        if self.layer not in self.viewer.layers or self._extent is None:
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
        
        self.utils_widget = UtilWidge(self.viewer)
        
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
        
        self.viewer.dims.events.current_step.connect(self._run_point_update)
        self.viewer_model3.dims.events.current_step.connect(self._run_point_update)
        
        self.viewer.dims.events.order.connect(self._order_update)
        self.viewer.events.reset_view.connect(self._reset_view)
        self.viewer_model1.events.status.connect(self._status_update)
        self.viewer_model2.events.status.connect(self._status_update)
        self.viewer_model3.events.status.connect(self._status_update)
        
        # 连接信号到槽
        self.message_signal.connect(self.utils_widget.print_in_widget)
    
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.refresh_timer = None

    
    def print_in_widget(self, text):
        self.utils_widget.print_in_widget(text)
        

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


    def _run_point_update(self, event):
        worker = Worker(self.viewer, self.viewer_model1, self.viewer_model2, self.viewer_model3)
        worker.finished.connect(self._on_point_update_finished)
        self.executor.submit(worker.run, event)

    def _on_point_update_finished(self):
        pass


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
        new_layers = [
            copy_layer(event.value, "model1"),
            copy_layer(event.value, "model2"),
            copy_layer(event.value, "model3")
        ]
        self.viewer_model1.layers.insert(event.index, new_layers[0])
        self.viewer_model2.layers.insert(event.index, new_layers[1])
        self.viewer_model3.layers.insert(event.index, new_layers[2])
        for layer, model in zip(new_layers, [self.viewer_model1, self.viewer_model2, self.viewer_model3]):
            for name in get_property_names(event.value):
                getattr(event.value.events, name).connect(own_partial(self._property_sync, name))
            if isinstance(event.value, Labels):
                event.value.events.set_data.connect(self._set_data_refresh)
                layer.events.set_data.connect(self._set_data_refresh)
            if event.value.name != ".cross":
                layer.events.data.connect(self._sync_data)
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
        
        # 检查数据是否真的发生变化，如果没有变化则不进行同步
        data_changed = any(event.source.data != model.layers[event.source.name].data
                        for model in [self.viewer, self.viewer_model1, self.viewer_model2, self.viewer_model3])
        if not data_changed:
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
        
        if self.refresh_timer is not None:
            self.refresh_timer.cancel()
        
        # 设置一个延迟刷新
        self.refresh_timer = threading.Timer(0.5, lambda: self._perform_refresh(event))
        self.refresh_timer.start()

    def _perform_refresh(self, event):
        layers_to_refresh = []
        
        for model in [self.viewer, self.viewer_model1, self.viewer_model2, self.viewer_model3]:
            if event.source == None:
                continue
            else:
                layer = model.layers[event.source.name]
                if layer is event.source:
                    continue
                
                if self._data_needs_refresh(layer):
                    layers_to_refresh.append(layer)
        
        try:
            self._block = True
            for layer in layers_to_refresh:
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
