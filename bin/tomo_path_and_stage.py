import os
from enum import Enum

class TomoPathAndStage:
    class ProgressStage(Enum):
        ISO_NET = "IsoNet"
        ISO_NET_DECONV = "IsoNet Deconvolution"
        ISO_NET_CORRECTION = "IsoNet Correction"
        MANUAL_CORRECTION = "Manual Correction"
        STATISTICAL_ANALYSIS = "Statistical Analysis"
        
        def __str__(self):
            return self.name.replace("_", " ")

    def __init__(self, current_path, pid):
        self.tomo_name = None
        self.current_path = current_path
        self.pid = pid
        self.root_dir = None

    def set_tomo_name(self, tomo_name):
        self.tomo_name = tomo_name
        self.root_dir = os.path.abspath(os.path.join(self.tomo_name, 'ves_seg', 'temp'))

        # 检查是否包含 `*-1`，并获取基本路径
        base_tomo_name = tomo_name.split('-1')[0] if '-1' in tomo_name else tomo_name

        # 初始化路径
        self.ori_tomo_path = os.path.abspath(os.path.join(tomo_name, base_tomo_name + '-bin4-wbp.rec'))
        self.tomograms_star_path = os.path.abspath(os.path.join(tomo_name, 'ves_seg', 'tomograms.star'))
        self.deconv_tomo_path = os.path.abspath(os.path.join(tomo_name, 'ves_seg', 'tomoset', base_tomo_name + '_dec.mrc'))
        self.correction_output_path = os.path.abspath(os.path.join(tomo_name, 'ves_seg'))
        self.isonet_tomo_path = os.path.abspath(os.path.join(tomo_name, 'ves_seg', base_tomo_name + '_wbp_corrected.mrc'))
        self.segment_path = os.path.abspath(os.path.join(tomo_name, 'ves_seg', base_tomo_name + '_segment.mrc'))
        self.label_path = os.path.abspath(os.path.join(tomo_name, 'ves_seg', base_tomo_name + '_label_vesicle.mrc'))
        self.json_file_path = os.path.abspath(os.path.join(tomo_name, 'ves_seg', base_tomo_name + '_vesicle.json'))
        self.new_json_file_path = os.path.join(self.root_dir, 'vesicle_new_{}.json'.format(self.pid))
        self.new_label_file_path = os.path.join(self.root_dir, 'label_{}.mrc'.format(self.pid))
        self.ori_json_file_path = os.path.abspath(os.path.join(tomo_name, 'ves_seg', base_tomo_name + '_vesicle_ori.json'))
        self.ori_label_path = os.path.abspath(os.path.join(tomo_name, 'ves_seg', base_tomo_name + '_label_vesicle_ori.mrc'))
        # 初始化进度
        self.progress_stage = self.determine_progress()
    
    def determine_progress(self):
        """确定当前的进度阶段"""
        if os.path.exists(self.ori_label_path):
            return self.ProgressStage.STATISTICAL_ANALYSIS
        elif os.path.exists(self.ori_json_file_path) or os.path.exists(self.new_label_file_path):
            return self.ProgressStage.MANUAL_CORRECTION
        elif os.path.exists(self.isonet_tomo_path):
            return self.ProgressStage.ISO_NET_CORRECTION
        elif os.path.exists(self.deconv_tomo_path):
            return self.ProgressStage.ISO_NET_DECONV
        else:
            return self.ProgressStage.ISO_NET

    def __str__(self):
        return f"TomoPath: {self.tomo_name}, Progress Stage: {self.progress_stage}"

# # 全局变量
# TOMO_SEGMENTATION_PROGRESS = ProgressStage.ISO_NET
# global_viewer = napari.Viewer()

if __name__ == '__main__':
    # 示例使用
    tomo_path_instance = TomoPathAndStage(tomo_name="example_tomo", root_dir="/path/to/root", pid=1234)
    print(tomo_path_instance)