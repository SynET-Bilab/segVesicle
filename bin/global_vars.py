from enum import Enum
import os

class ProgressStage(Enum):
    ISO_NET = 1
    ISO_NET_DECONV = 1.1
    ISO_NET_CORRECTION = 1.2
    IMAGE_SEGMENTATION = 2
    MANUAL_CORRECTION = 3
    STATISTICAL_ANALYSIS = 4
    
    def __str__(self):
        return self.name.replace("_", " ")

class TomoPath:
    def __init__(self, tomo_dir, root_dir, pid):
        self.tomo_dir = tomo_dir
        self.root_dir = root_dir
        self.pid = pid

        # 初始化路径
        # self.ori_tomo_path = os.path.abspath(tomo_dir + '_wbp.mrc')
        self.ori_tomo_path = os.path.abspath(os.path.join(os.pardir, tomo_dir + '-bin4-wbp.rec'))
        self.deconv_tomo_path = os.path.abspath('tomoset/' + tomo_dir + '_dec.mrc')
        self.isonet_tomo_path = os.path.abspath(tomo_dir + '_wbp_corrected.mrc')
        self.segment_path = os.path.abspath(tomo_dir + '_segment.mrc')
        self.label_path = os.path.abspath(tomo_dir + '_label_vesicle.mrc')
        self.json_file_path = os.path.abspath(tomo_dir + '_vesicle.json')
        self.new_json_file_path = os.path.join(root_dir, 'vesicle_new_{}.json'.format(pid))
        self.new_label_file_path = os.path.join(root_dir, 'label_{}.mrc'.format(pid))
        self.ori_json_file_path = os.path.abspath(tomo_dir + '_vesicle_ori.json')
        self.ori_label_path = os.path.abspath(tomo_dir + '_label_vesicle_ori.mrc')

        # 初始化进度
        self.determine_progress()
    
    def determine_progress(self):
        """确定当前的进度阶段"""
        global TOMO_SEGMENTATION_PROGRESS
        
        if os.path.exists(self.ori_label_path):
            TOMO_SEGMENTATION_PROGRESS = ProgressStage.STATISTICAL_ANALYSIS
        elif os.path.exists(self.ori_json_file_path) or os.path.exists(self.new_label_file_path):
            TOMO_SEGMENTATION_PROGRESS = ProgressStage.MANUAL_CORRECTION
        elif os.path.exists(self.isonet_tomo_path):
            TOMO_SEGMENTATION_PROGRESS = ProgressStage.ISO_NET_CORRECTION
        elif os.path.exists(self.deconv_tomo_path):
            TOMO_SEGMENTATION_PROGRESS = ProgressStage.ISO_NET_DECONV
        else:
            TOMO_SEGMENTATION_PROGRESS = ProgressStage.ISO_NET

# 全局变量
TOMO_SEGMENTATION_PROGRESS = ProgressStage.ISO_NET
global_viewer = None