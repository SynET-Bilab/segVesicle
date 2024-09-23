import napari
import sys
import logging
import mrcfile
import os
import subprocess
import numpy as np
from tqdm import tqdm
from qtpy.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QApplication, QTextEdit, QProgressDialog, QMessageBox
from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon

from util.add_layer_with_right_contrast import add_layer_with_right_contrast
from util.model_exists import ensure_model_exists

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

class CorrectionWindow(QMainWindow):
    def __init__(self, tomo_viewer):        
        super().__init__(tomo_viewer.viewer.window.qt_viewer)
        self.setWindowTitle('Correction')
        self.setGeometry(100, 100, 800, 400)

        # Main layout
        layout = QVBoxLayout()

        # New button for 'create tomo star'
        self.create_tomo_star_button = QPushButton('Create tomo star')
        layout.addWidget(self.create_tomo_star_button)
        self.create_tomo_star_button.clicked.connect(self.create_tomo_star)


        # First button
        self.button1 = QPushButton('Correction in Vesicle Segmentation (about 10 minutes)')
        layout.addWidget(self.button1)
        self.button1.clicked.connect(self.correction)

        # Second button
        self.button2 = QPushButton('Correction in Terminal (Recommended)')
        layout.addWidget(self.button2)
        self.button2.clicked.connect(self.copy_code_and_close)

        # Code display and copy button
        code_layout = QHBoxLayout()
        self.tomo_viewer = tomo_viewer
        tomograms_star_path = tomo_viewer.tomo_path_and_stage.tomograms_star_path
        output_path = tomo_viewer.tomo_path_and_stage.correction_output_path
        # self.model_path = '/share/data/CryoET_Data/lvzy/script/segvesicle/segvesv0.1/vesicle_corrected_model.h5'
        self.model_name = 'vesicle_corrected_model.h5'
        self.model_path = ensure_model_exists(self.model_name)
        if self.model_path is None:
            line = f'isonet.py predict {tomograms_star_path} --output_dir {output_path} model_path --gpuID 0'
        else:
            line = f'isonet.py predict {tomograms_star_path} --output_dir {output_path} {self.model_path} --gpuID 0'
        
        # 代码显示和复制按钮
        code_layout = QHBoxLayout()
        self.code_text = QTextEdit()
        self.code_text.setReadOnly(True)
        self.code_text.setPlainText(line)
        code_layout.addWidget(self.code_text)

        self.copy_button = QPushButton()
        self.copy_button.setIcon(QIcon.fromTheme('edit-copy'))  # 使用标准复制图标主题
        self.copy_button.clicked.connect(self.copy_code_to_clipboard)
        code_layout.addWidget(self.copy_button)

        layout.addLayout(code_layout)
        # Set layout
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
    def create_tomo_star(self):
        """
        Function to execute the 'isonet.py prepare_star' command with updated paths.
        """
        try:
            # 获取 rec_tomo_path 所在的文件夹路径
            rec_tomo_dir = os.path.dirname(self.tomo_viewer.tomo_path_and_stage.rec_tomo_path)
            
            # 获取 tomograms_star_path
            output_star = self.tomo_viewer.tomo_path_and_stage.tomograms_star_path
            
            # 构建命令
            command = [
                'isonet.py', 'prepare_star', rec_tomo_dir,
                '--pixel_size', '17.14',
                '--output_star', output_star
            ]
            
            # 执行命令
            subprocess.run(command, check=True)
            
            # 成功提示
            self.tomo_viewer.print('Tomo star created successfully!')
        
        except subprocess.CalledProcessError as e:
            # 错误提示
            self.tomo_viewer.print('Failed to create tomo star.')

    def copy_code_to_clipboard(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.code_text.toPlainText())
        
    def copy_code_and_close(self):
        self.copy_code_to_clipboard()
        self.close()
        
    def correction(self):
        if not TF_AVAILABLE:
            self.tomo_viewer.print("TensorFlow is not available. Correction cannot be performed.")
            self.close()
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("TensorFlow Not Available")
            msg_box.setText("TensorFlow is not available. Correction cannot be performed.")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()
            return
        
        self.progress_dialog = QProgressDialog("Processing...", 'Cancel', 0, 100, self)
        self.progress_dialog.setWindowTitle('Correcting')
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setValue(0)
        self.progress_dialog.show()
        
        data = self.tomo_viewer.viewer.layers['deconv_tomo'].data
        self.progress_dialog.setValue(50)
        
        correction_data = self.predict_one(data)
        directory = os.path.dirname(self.tomo_viewer.tomo_path_and_stage.isonet_tomo_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        
        # 将数据缩放到 -128 到 127 的范围
        correction_data_normalized = (correction_data - correction_data.min()) / (correction_data.max() - correction_data.min())

        correction_data_int8 = (correction_data_normalized * 255 - 128).astype(np.int8)

        with mrcfile.new(self.tomo_viewer.tomo_path_and_stage.isonet_tomo_path, overwrite=True) as output_mrc:
            output_mrc.set_data(correction_data_int8)
            output_mrc.voxel_size = 17.14
        
        add_layer_with_right_contrast(correction_data, 'corrected_tomo', self.tomo_viewer.viewer)
        self.tomo_viewer.viewer.layers.move(self.tomo_viewer.viewer.layers.index(self.tomo_viewer.viewer.layers['corrected_tomo']), 0)
        self.tomo_viewer.viewer.layers['deconv_tomo'].visible = False
        self.tomo_viewer.viewer.layers.selection.active = self.tomo_viewer.viewer.layers['edit vesicles']
        self.tomo_viewer.print("Finish Correction.")
        self.progress_dialog.setValue(100)
        self.tomo_viewer.show_current_state()
        self.close()

    def predict_one(self, one_tomo):
        model = tf.keras.models.load_model(self.model_path)

        # root_name = one_tomo.split('/')[-1].split('.')[0]

        real_data = self.tomo_viewer.viewer.layers['deconv_tomo'].data.astype(np.float32) * -1
        data = normalize(real_data, percentile=True)

        reform_ins = reform3D(data, 64, 96, 9)
        data = reform_ins.pad_and_crop()

        N = 4
        num_patches = data.shape[0]
        if num_patches % N == 0:
            append_number = 0
        else:
            append_number = N - num_patches % N
        data = np.append(data, data[0:append_number], axis=0)
        num_big_batch = data.shape[0] // N
        outData = np.zeros(data.shape)
        logging.info("total batches: {}".format(num_big_batch))
        for i in tqdm(range(num_big_batch), file=sys.stdout):
            in_data = data[i * N:(i + 1) * N]
            outData[i * N:(i + 1) * N] = model.predict(in_data, verbose=0).squeeze()
        outData = outData[0:num_patches]
        outData = reform_ins.restore(outData)
        outData = normalize(outData, percentile=True)
        outData = outData * -1

        return outData
    
def normalize(x, percentile = True, pmin=4.0, pmax=96.0, axis=None, clip=False, eps=1e-20):
    """Percentile-based image normalization."""

    if percentile:
        mi = np.percentile(x,pmin,axis=axis,keepdims=True)
        ma = np.percentile(x,pmax,axis=axis,keepdims=True)
        out = (x - mi) / ( ma - mi + eps )
        out = out.astype(np.float32)
        if clip:
            return np.clip(out,0,1)
        else:
            return out
    else:
        out = (x-np.mean(x))/np.std(x)
        out = out.astype(np.float32)
        return out

class reform3D:
    def __init__(self,data3D, cubesize, cropsize, edge_depth):
        self._sp = np.array(data3D.shape)
        self._orig_data = data3D
        self.cubesize = cubesize
        self.cropsize = cropsize
        self.edge_depth = edge_depth
        self._sidelen = np.ceil((self._sp + edge_depth * 2)/self.cubesize).astype(int)
        #self._sidelen = np.ceil((1.*self._sp)/self.cubesize).astype(int)

    def pad_and_crop(self):
        
        #----------------------------|---------------------------
        #|                           |
        #|     ---------------|------|--------edge---------------
        #|     |  ------------|------|--------image_edge----------  
        #|     |  |           |      |
        #|     |  |           |      |
        #|     |  |           |      |
        #|     |  |           |      |
        #|     -----cube-------      | 
        #|     |  |                  |
        #|     |  |                  |
        #|----------crop--------------       
        #|     |  |        
        #|     |  |       
        #|     |  |       
        #|     |  |
        pad_left = int((self.cropsize - self.cubesize)/2 + self.edge_depth)
        
        # pad_right + pad_left + shape = sidelen * cube_zize + (crop_size-cube_size)
        # pad_right + pad_left + shape >= (self._sp + edge_depth * 2) + (crop_size-cube_size)
        pad_right = (self._sidelen * self.cubesize + (self.cropsize-self.cubesize) - pad_left - self._sp).astype(int)

        data = np.pad(self._orig_data,((pad_left,pad_right[0]),(pad_left,pad_right[1]),(pad_left,pad_right[2])),'symmetric')
        outdata=[]

        for i in range(self._sidelen[0]):
            for j in range(self._sidelen[1]):
                for k in range(self._sidelen[2]):
                    cube = data[i*self.cubesize:i*self.cubesize+self.cropsize,
                            j*self.cubesize:j*self.cubesize+self.cropsize,
                            k*self.cubesize:k*self.cubesize+self.cropsize]
                    outdata.append(cube)
        outdata=np.array(outdata)
        return outdata
    
    def mask(self, x_len, y_len, z_len):
        # need to consider should partisioned to len+1 so that left and right can add to one
        p = 2*self.edge_depth#(self.cropsize - self.cubesize)
        assert x_len > 2*p
        assert y_len > 2*p
        assert z_len > 2*p

        array_x = np.minimum(np.arange(x_len+1), p) / p
        array_x = array_x * np.flip(array_x)
        array_x  = array_x[np.newaxis,np.newaxis,:]

        array_y = np.minimum(np.arange(y_len+1), p) / p
        array_y = array_y * np.flip(array_y)
        array_y  = array_y[np.newaxis,:,np.newaxis]

        array_z = np.minimum(np.arange(z_len+1), p) / p
        array_z = array_z * np.flip(array_z)
        array_z  = array_z[:,np.newaxis,np.newaxis]

        out = array_x * array_y * array_z
        return out[:x_len,:y_len,:z_len]


    def restore(self,cubes):

        start = (self.cropsize-self.cubesize)//2-self.edge_depth
        end = (self.cropsize-self.cubesize)//2+self.cubesize+self.edge_depth
        cubes = cubes[:,start:end,start:end,start:end]

        restored = np.zeros((self._sidelen[0]*self.cubesize+self.edge_depth*2,
                        self._sidelen[1]*self.cubesize+self.edge_depth*2,
                        self._sidelen[2]*self.cubesize+self.edge_depth*2))
        print("size restored", restored.shape)
        mask_cube = self.mask(self.cubesize+self.edge_depth*2,self.cubesize+self.edge_depth*2,self.cubesize+self.edge_depth*2)
        for i in range(self._sidelen[0]):
            for j in range(self._sidelen[1]):
                for k in range(self._sidelen[2]):
                    restored[i*self.cubesize:(i+1)*self.cubesize+self.edge_depth*2,
                        j*self.cubesize:(j+1)*self.cubesize+self.edge_depth*2,
                        k*self.cubesize:(k+1)*self.cubesize+self.edge_depth*2] \
                        += cubes[i*self._sidelen[1]*self._sidelen[2]+j*self._sidelen[2]+k]\
                            *mask_cube
                        
                    
        p =self.edge_depth*2 #int((self.cropsize-self.cubesize)/2+self.edge_depth)
        restored = restored[p:p+self._sp[0],p:p+self._sp[1],p:p+self._sp[2]]
        return restored

    def mask_old(self):
        from functools import reduce
        c = self.cropsize
        p = (self.cropsize - self.cubesize)
        mask = np.ones((c, c, c))
        f = lambda x: min(x, p)/p 
        for i in range(c):
            for j in range(c):
                for k in range(c):
                    d = [i, c-i, j, c-j, k, c-k]
                    d = map(f,d)
                    d = reduce(lambda a,b: a*b, d)
                    mask[i,j,k] = d
        return mask
    def restore_from_cubes(self,cubes):

        new = np.zeros((self._sidelen[0]*self.cubesize,
                        self._sidelen[1]*self.cubesize,
                        self._sidelen[2]*self.cubesize))
        start=int((self.cropsize-self.cubesize)/2)
        end=int((self.cropsize+self.cubesize)/2)
        
        for i in range(self._sidelen[0]):
            for j in range(self._sidelen[1]):
                for k in range(self._sidelen[2]):
                    new[i*self.cubesize:(i+1)*self.cubesize,
                        j*self.cubesize:(j+1)*self.cubesize,
                        k*self.cubesize:(k+1)*self.cubesize] \
                        = cubes[i*self._sidelen[1]*self._sidelen[2]+j*self._sidelen[2]+k][start:end,start:end,start:end]
        return new[0:self._sp[0],0:self._sp[1],0:self._sp[2]]
    def pad4times(self,time=4):
        sp = np.array(self._orig_data.shape)
        sp = np.expand_dims(sp,axis=0)
        padsize = (sp // time + 1) * time - sp
        self._padsize =padsize
        print(padsize, np.zeros((len(self._orig_data.shape),1)))
        width = np.concatenate((np.zeros((len(self._orig_data.shape),1),int),padsize.T),axis=1)
        return np.pad(self._orig_data,width,'edge')
    def cropback(self,padded):
        sp = padded.shape
        ps = self._padsize
        orig_sp = sp - ps
        print ('orig_sp',orig_sp)
        return padded[:orig_sp[0][0]][:orig_sp[0][1]][:orig_sp[0][2]]