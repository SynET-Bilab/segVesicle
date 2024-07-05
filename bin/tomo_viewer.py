import numpy as np
import mrcfile
import napari

from qtpy import QtCore, QtWidgets


from napari.utils.notifications import show_info

from three_orthos_viewer import CrossWidget, MultipleViewerWidget
from tomo_path_and_stage import TomoPathAndStage

from window.deconv_window import DeconvWindow
from window.correction_window import CorrectionWindow
from util.deconvolution import deconv_tomo
from util.add_layer_with_right_contrast import add_layer_with_right_contrast



class TomoViewer:
    def __init__(self, viewer: napari.Viewer, current_path: str, pid: int):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
        self.viewer: napari.Viewer = viewer
        self.main_viewer = self.viewer.window.qt_viewer.parentWidget()
        self.multiple_viewer_widget: MultipleViewerWidget = MultipleViewerWidget(self.viewer)
        self.tomo_path_and_stage: TomoPathAndStage = TomoPathAndStage(current_path, pid)
        self.cross_widget: CrossWidget = CrossWidget(self.viewer)
        self.main_viewer.layout().addWidget(self.multiple_viewer_widget)
        self.viewer.window.add_dock_widget(self.cross_widget, name="Cross", area="left")
        
    def set_tomo_name(self, tomo_name: str):
        self.tomo_path_and_stage.set_tomo_name(tomo_name)
        
    def print(self, message):
        self.multiple_viewer_widget.print_in_widget(message)
        
    def register_isonet(self):
        self.register_correction_tomo()
        self.register_deconv_tomo()
        self.register_open_ori_tomo()
        
    def register_open_ori_tomo(self):
        def get_tomo(path):
            with mrcfile.open(path) as mrc:
                data = mrc.data
            data = np.flip(data, axis=1)
            return data
        def button_clicked():
            path = self.tomo_path_and_stage.ori_tomo_path
            data = get_tomo(path)
            
            add_layer_with_right_contrast(data, 'ori_tomo', self.viewer)
            
            self.viewer.layers['corrected_tomo'].visible = False
            self.viewer.layers.move(self.viewer.layers.index(self.viewer.layers['ori_tomo']), 0)
            self.viewer.layers.selection.active = self.viewer.layers['edit vesicles']
        try:
            self.multiple_viewer_widget.utils_widget.ui.open_bin4wbp.clicked.disconnect()
        except TypeError:
            pass

        self.multiple_viewer_widget.utils_widget.ui.open_bin4wbp.clicked.connect(button_clicked)
        
    def register_deconv_tomo(self):
        def open_deconv_window():
            if 'ori_tomo' in self.viewer.layers:
                if len(self.viewer.layers['edit vesicles'].data) == 2:
                    self.deconv_window = DeconvWindow(self.viewer)
                    self.deconv_window.show()
                else:
                    self.print('Please add two points to define deconvolution area.')
                    show_info('Please add two points to define deconvolution area.')
            else:
                self.print('Please open original tomo.')
                show_info('Please open original tomo.')
        self.multiple_viewer_widget.utils_widget.ui.deconvolution.clicked.connect(open_deconv_window)
        
    def register_correction_tomo(self):
        def open_correction_window():
            if 'deconv_tomo' in self.viewer.layers:
                self.correction_window = CorrectionWindow(self)
                self.correction_window.show()
            else:
                self.print('Please perform deconvolution.')
                show_info('Please perform deconvolution.')
        self.multiple_viewer_widget.utils_widget.ui.correction.clicked.connect(open_correction_window)