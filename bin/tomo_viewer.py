import numpy as np
import mrcfile
import napari
import time

from qtpy import QtCore, QtWidgets
from tqdm import tqdm

from skimage.measure import label
from napari.utils.notifications import show_info
from skimage.morphology import opening, closing, erosion, dilation, remove_small_objects
from skimage.morphology import cube, ball, disk, square
from scipy.sparse import csr_matrix
from three_orthos_viewer import CrossWidget, MultipleViewerWidget
from tomo_path_and_stage import TomoPathAndStage

from segVesicle.bin.boundary_mask import boundary_mask
from window.deconv_window import DeconvWindow
from window.correction_window import CorrectionWindow
from util.add_layer_with_right_contrast import add_layer_with_right_contrast
from segVesicle.models import resunet3D as models




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
        self.multiple_viewer_widget.utils_widget.ui.finish_isonet.clicked.connect(self.on_finish_isonet_clicked)
        self.multiple_viewer_widget.utils_widget.ui.predict.clicked.connect(self.predict_clicked)
        
    def register_open_ori_tomo(self):
        def get_tomo(path):
            with mrcfile.open(path) as mrc:
                data = mrc.data
            data = np.flip(data, axis=1)
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
        
    def on_finish_isonet_clicked(self):
        self.multiple_viewer_widget.utils_widget.ui.tabWidget.setCurrentIndex(2)
        
    def predict_clicked(self):
        self.deconv_data = self.viewer.layers['deconv_tomo'].data
        self.corrected_data = self.viewer.layers['corrected_tomo'].data
        self.label = self.predict_label(self.deconv_data, self.corrected_data)
        self.area_path = self.tomo_path_and_stage.area_path
        self.processed_vesicles, self.shape = self.morph_process(self.label, self.area_path)
        self.viewer.add_labels(self.processed_vesicles, name='new_label')
        
    def predict_label(self, deconv_data, corrected_data):
        path_weights1 = '/home/liushuo/Documents/code/vesiclePipeline/segVesicle/pretrained/vesicle_seg_model_1.h5'
        path_weights2 = '/home/liushuo/Documents/code/vesiclePipeline/segVesicle/pretrained/vesicle_seg_model_2.h5'

        seg1 = self.segment(path_weights1, corrected_data)
        seg2 = self.segment(path_weights2, deconv_data)
        labelmap = np.sign(seg1 + seg2).astype(np.int8)
        
        return labelmap
    
    def segment(self, path_weights, data, patch_size=192):
        pcrop = 48  # how many pixels to crop from border

        P = patch_size
        Ncl = 2
        # build network
        net = models.my_model(patch_size, Ncl)
        net.load_weights(path_weights)

        percentile_99_5 = np.percentile(data, 99.5)
        percentile_00_5 = np.percentile(data, 00.5)
        data = np.clip(data, percentile_00_5, percentile_99_5)

        data = (data - np.mean(data)) / np.std(data)  # normalize
        data = np.pad(data, pcrop, mode='constant', constant_values=0)  # 0pad
        dim = data.shape
        l = int(P / 2)
        lcrop = int(l - pcrop)
        step = int(2 * l - 2 * pcrop)
        # Get patch centers:
        pcenterZ = list(range(l, dim[0] - l, step))
        pcenterY = list(range(l, dim[1] - l, step))
        pcenterX = list(range(l, dim[2] - l, step))
        # If there are still few pixels at the end:
        if pcenterX[-1] < dim[2] - l:
            pcenterX.append(dim[2] - l)
        if pcenterY[-1] < dim[1] - l:
            pcenterY.append(dim[1] - l)
        if pcenterZ[-1] < dim[0] - l:
            pcenterZ.append(dim[0] - l)
        Npatch = len(pcenterX) * len(pcenterY) * len(pcenterZ)
        print('Data array is divided in ' + str(Npatch) + ' patches ...')
        # ---------------------------------------------------------------
        # Process data in patches:
        start = time.time()

        predArray = np.zeros(dim + (Ncl,), dtype=np.float16)
        normArray = np.zeros(dim, dtype=np.int8)
        patchCount = 1
        for x in pcenterX:
            for y in pcenterY:
                for z in pcenterZ:
                    print(f'Segmenting patch {patchCount} / {Npatch} ...')
                    patch = data[z - l:z + l, y - l:y + l, x - l:x + l]
                    patch = np.reshape(patch, (1, P, P, P, 1))  # reshape for keras [batch,x,y,z,channel]
                    pred = net.predict(patch, batch_size=10)
                    predArray[z-lcrop:z+lcrop, y-lcrop:y+lcrop, x-lcrop:x+lcrop, :] += np.float16(pred[0, l - lcrop:l + lcrop, l - lcrop:l + lcrop, l - lcrop:l + lcrop, :])
                    normArray[z-lcrop:z+lcrop, y-lcrop:y+lcrop, x-lcrop:x+lcrop] += np.ones((P-2*pcrop, P-2*pcrop, P-2*pcrop), dtype=np.int8)
                    patchCount += 1
        normArray[normArray == 0] = 1

        # Normalize overlapping regions:
        for C in range(0, Ncl):
            predArray[:, :, :, C] = predArray[:, :, :, C] / normArray
        end = time.time()
        print(f"Model took {int(end - start)} seconds to predict")
        predArray = predArray[pcrop:-pcrop, pcrop:-pcrop, pcrop:-pcrop, :]  # unpad

        labelmap = np.int8(np.argmax(predArray, 3))
        return labelmap
    
    def morph_process(self, labelmap, area_file, pixelsize=17.14, radius=10):
        # 1. closing and opening process of vesicle mask. 2. label the vesicles.
        # 3. exclude false vesicles by counting their volumes and thresholding, return only vesicle binary mask
        # 4. extract boundaries and labels them
        # 5. extract labeled individual vesicle boundary, convert into points vectors and output them.
        tomo_mask = labelmap.astype(np.int8)
        area_mask = boundary_mask(tomo_mask, area_file, pixelsize)
        tomo_mask *= area_mask
        
        # transform mask into uint8
        bimask = np.round(tomo_mask).astype(np.uint8)
        shape = bimask.shape

        # extract labeled mask whose area more than a threshold 
        # (just after prediction, some vesicles will be predicted to be connected)
        area_thre = radius**3
        labeled_pre = label(bimask)
        sup_pro = np.zeros(labeled_pre.shape)
        pre_pro = np.zeros(labeled_pre.shape)
        idx_pre = self.get_indices_sparse(labeled_pre)
        num_pre = np.max(labeled_pre)

        print('\nFirst separation of the mask by volume thresholding\n')
        for i in tqdm(range(1, num_pre+1)):
            if idx_pre[i][0].shape[0] > area_thre*15:
                pre_pro[idx_pre[i][0], idx_pre[i][1], idx_pre[i][2]] = 1
                labeled_pre[idx_pre[i][0], idx_pre[i][1], idx_pre[i][2]] = 0
        labeled_pre[labeled_pre > 1] = 1

        kernel_pre = cube(11)
        pre_pro = opening(pre_pro, kernel_pre)
        pre_pro = erosion(pre_pro, cube(2))
        labeled_pre_pro = label(pre_pro) #process linked vesicles just after prediction, Part 1

        print('\nFix the broken vesicles\n')
        kernel_xy = np.reshape([1 for i in range(9)], (3, 3, 1))
        closing_opening_xy = opening(labeled_pre, kernel_xy)
        kernel = np.reshape([1 for i in range(12)], (2, 2, 3))
        closing_opening = opening(closing_opening_xy, kernel)

        labeled = label(closing_opening)
        post_pro = np.zeros(labeled.shape)
        idx = self.get_indices_sparse(labeled)
        num = np.max(labeled)

        print('\nSecond separation of the mask by volume thresholding\n')
        for i in tqdm(range(1, num+1)):
            if idx[i][0].shape[0] < area_thre:
                labeled[idx[i][0], idx[i][1], idx[i][2]] = 0
                if idx[i][0].shape[0] > 0.2*area_thre:
                    sup_pro[idx[i][0], idx[i][1], idx[i][2]] = 1
            elif idx[i][0].shape[0] <= 2*area_thre and idx[i][0].shape[0] > area_thre:
                sup_pro[idx[i][0], idx[i][1], idx[i][2]] = 1
            elif idx[i][0].shape[0] > area_thre*12:
                post_pro[idx[i][0], idx[i][1], idx[i][2]] = 1
                labeled[idx[i][0], idx[i][1], idx[i][2]] = 0

        labeled = label(labeled)
        num = np.max(labeled)

        kernel_p = cube(5)
        post_pro = opening(post_pro, kernel_p)
        labeled_post_pro = label(post_pro)
        num_post = np.max(labeled_post_pro)

        labeled_post_pro += num
        labeled_post_pro[labeled_post_pro == num] = 0

        num += num_post
        labeled_pre_pro += num
        labeled_pre_pro[labeled_pre_pro == num] = 0
        labeled = labeled + labeled_post_pro + labeled_pre_pro
        num = np.max(labeled)

        filtered = (labeled >= 1).astype(np.uint8)
        print('\ncomplete filtering\n')
        boundaries = filtered - erosion(filtered, cube(3))
        bd_labeled = label(boundaries)
        idx = self.get_indices_sparse(bd_labeled)
        vesicle_list = [np.swapaxes(np.array(idx[i]), 0, 1) for i in range(1, np.max(bd_labeled)+1)]
        
        return vesicle_list, shape

    def get_indices_sparse(self, data):
        M = self.compute_M(data)
        return [np.unravel_index(row.data, data.shape) for row in M]

    def compute_M(self, data):
        cols = np.arange(data.size)
        return csr_matrix((cols, (data.ravel(), cols)), shape=(data.max()+1, data.size))
