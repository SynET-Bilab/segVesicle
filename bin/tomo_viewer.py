import numpy as np
import mrcfile
import napari
import os
import subprocess

from qtpy import QtCore, QtWidgets
from napari.utils.notifications import show_info

from three_orthos_viewer import CrossWidget, MultipleViewerWidget
from tomo_path_and_stage import TomoPathAndStage

from window.deconv_window import DeconvWindow
from window.correction_window import CorrectionWindow
from util.add_layer_with_right_contrast import add_layer_with_right_contrast
from util.predict_vesicle import predict_label, morph_process, vesicle_measure, vesicle_rendering




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
        self.multiple_viewer_widget.utils_widget.ui.finish_isonet.clicked.disconnect()
        self.multiple_viewer_widget.utils_widget.ui.finish_isonet.clicked.connect(self.on_finish_isonet_clicked)
        self.multiple_viewer_widget.utils_widget.ui.predict.clicked.disconnect()
        self.multiple_viewer_widget.utils_widget.ui.predict.clicked.connect(self.predict_clicked)
        self.register_draw_area_mod()
        
    def register_open_ori_tomo(self):
        def get_tomo(path):
            with mrcfile.open(path) as mrc:
                data = mrc.data
            return data
        def button_clicked():
            from qtpy.QtWidgets import QProgressDialog
            from qtpy.QtCore import Qt
            self.progress_dialog = QProgressDialog("Processing...", 'Cancel', 0, 100, self.main_viewer)
            self.progress_dialog.setWindowTitle('Opening')
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.setValue(0)
            self.progress_dialog.show()
            path = self.tomo_path_and_stage.ori_tomo_path
            data = get_tomo(path)
            self.progress_dialog.setValue(50)
            add_layer_with_right_contrast(data, 'ori_tomo', self.viewer)
            
            self.viewer.layers['corrected_tomo'].visible = False
            self.viewer.layers.move(self.viewer.layers.index(self.viewer.layers['ori_tomo']), 0)
            self.viewer.layers.selection.active = self.viewer.layers['edit vesicles']
            self.progress_dialog.setValue(100)
            message = f"Successfully opened the original image {self.tomo_path_and_stage.ori_tomo_path}."
            self.print(message)
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
        self.multiple_viewer_widget.utils_widget.ui.deconvolution.clicked.disconnect()
        self.multiple_viewer_widget.utils_widget.ui.deconvolution.clicked.connect(open_deconv_window)
        
    def register_correction_tomo(self):
        def open_correction_window():
            if 'deconv_tomo' in self.viewer.layers:
                self.correction_window = CorrectionWindow(self)
                self.correction_window.show()
            else:
                self.print('Please perform deconvolution.')
                show_info('Please perform deconvolution.')
        self.multiple_viewer_widget.utils_widget.ui.correction.clicked.disconnect()
        self.multiple_viewer_widget.utils_widget.ui.correction.clicked.connect(open_correction_window)
        
    def on_finish_isonet_clicked(self):
        self.multiple_viewer_widget.utils_widget.ui.tabWidget.setCurrentIndex(2)
        
    def predict_clicked(self):
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
        self.ves_tomo = vesicle_rendering(self.vesicle_info, self.shape)
        
        self.viewer.add_labels(self.ves_tomo, name='new_label')
        self.progress_dialog.setValue(100)
        
    def register_draw_area_mod(self):
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
                self.print("Points validated and saved successfully.")
            else:
                self.print("Points do not meet the required conditions.")
        self.multiple_viewer_widget.utils_widget.ui.draw_tomo_area.clicked.disconnect()
        self.multiple_viewer_widget.utils_widget.ui.draw_tomo_area.clicked.connect(create_area_mod)
      
    
        
def validate_points(points):
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