import os
from enum import Enum

class TomoPathAndStage:
    class ProgressStage(Enum):
        SELECT_TOMO = "Please Select Tomo"
        OPEN_TOMO = "Please Open Original Tomo"
        MAKE_DECONVOLUTION = "Please Make Deconvolution"
        MAKE_CORRECTION = "Please Make Correction"
        MAKE_PREDICT = "Please Make Prediction"
        MANUALLY_CORRECT_LABELS = "Please Manually Correct Labels"

    def __init__(self, current_path, pid):
        self.tomo_name = None
        self.current_path = current_path
        self.pid = pid
        self.root_dir = None
        
        # 初始化进度
        self.progress_stage = self.determine_progress()

    def set_tomo_name(self, tomo_name):
        self.tomo_name = tomo_name
        self.root_dir = os.path.join(self.current_path, self.tomo_name, 'ves_seg', 'temp')

        # 检查是否包含 `*-1`，并获取基本路径
        self.base_tomo_name = tomo_name.split('-1')[0] if '-1' in tomo_name else tomo_name

        # 初始化路径
        # self.ori_tomo_path = os.path.join(self.current_path, tomo_name, base_tomo_name + '-bin4-wbp.rec')
        self.ori_tomo_path = os.path.join(self.current_path, tomo_name, self.base_tomo_name + '-bin4-5i.rec')
        self.rec_tomo_path = os.path.join(self.current_path, tomo_name, 'ves_seg', 'tomoset', self.base_tomo_name + '_wbp_resample.mrc')
        self.tomograms_star_path = os.path.join(self.current_path, tomo_name, 'ves_seg', 'tomograms.star')
        self.deconv_tomo_path = os.path.join(self.current_path, tomo_name, 'ves_seg', 'tomo_deconv', self.base_tomo_name + '_wbp_resample.mrcc')
        self.correction_output_path = os.path.join(self.current_path, tomo_name, 'ves_seg')
        self.area_path = os.path.join(self.current_path, tomo_name, 'area.mod')
        self.memb_prompt_path = self.tomograms_star_path = os.path.join(self.current_path, tomo_name, 'ves_seg', 'membrane', 'prompt.mod')
        self.memb_folder_path = self.tomograms_star_path = os.path.join(self.current_path, tomo_name, 'ves_seg', 'membrane')
        self.memb_result_path = self.tomograms_star_path = os.path.join(self.current_path, tomo_name, 'ves_seg', 'membrane', self.base_tomo_name + '.mod')
        self.isonet_tomo_path = os.path.join(self.current_path, tomo_name, 'ves_seg', self.base_tomo_name + '_wbp_corrected.mrc')
        self.segment_path = os.path.join(self.current_path, tomo_name, 'ves_seg', self.base_tomo_name + '_segment.mrc')
        self.label_path = os.path.join(self.current_path, tomo_name, 'ves_seg', self.base_tomo_name + '_label_vesicle.mrc')
        self.json_file_path = os.path.join(self.current_path, tomo_name, 'ves_seg', self.base_tomo_name + '_vesicle.json')
        self.new_json_file_path = os.path.join(self.root_dir, 'vesicle_new_{}.json'.format(self.pid))
        self.new_label_file_path = os.path.join(self.root_dir, 'label_{}.mrc'.format(self.pid))
        self.ori_json_file_path = os.path.join(self.current_path, tomo_name, 'ves_seg', self.base_tomo_name + '_vesicle_ori.json')
        self.ori_label_path = os.path.join(self.current_path, tomo_name, 'ves_seg', self.base_tomo_name + '_label_vesicle_ori.mrc')

        # 初始化进度
        self.progress_stage = self.determine_progress()
    
    def determine_progress(self):
        if self.tomo_name == None:
            return self.ProgressStage.SELECT_TOMO
        """确定当前的进度阶段"""    
        if os.path.exists(self.new_label_file_path) or os.path.exists(self.label_path):
            return self.ProgressStage.MANUALLY_CORRECT_LABELS
        elif os.path.exists(self.isonet_tomo_path):
            return self.ProgressStage.MAKE_PREDICT
        elif os.path.exists(self.deconv_tomo_path):
            return self.ProgressStage.MAKE_CORRECTION
        elif os.path.exists(self.ori_tomo_path):
            return self.ProgressStage.MAKE_DECONVOLUTION
        else:
            return self.ProgressStage.OPEN_TOMO
        
    def update_progress_stage(self):
        """更新进度阶段"""
        self.progress_stage = self.determine_progress()