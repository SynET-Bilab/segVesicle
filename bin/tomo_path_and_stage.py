import json
import os
import traceback
from datetime import datetime, timezone
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
        self.new_json_file_path = None
        self.new_label_file_path = None
        self.point_file_path = None
        
        # 初始化进度
        self.progress_stage = self.determine_progress()

    def set_tomo_name(self, tomo_name):
        self.tomo_name = tomo_name
        
        user_home_dir = os.path.expanduser("~")
        # 设置隐藏的临时存储路径
        self.root_dir = os.path.join(user_home_dir, '.segvesicle')
        # self.root_dir = os.path.join(self.current_path, self.tomo_name, 'ves_seg', 'temp')

        # 检查是否包含 `*-1`，并获取基本路径
        self.base_tomo_name = tomo_name.split('-1')[0] if '-1' in tomo_name else tomo_name

        # 初始化路径
        self.ori_tomo_path = None
        # self.ori_tomo_path = os.path.join(self.current_path, tomo_name, self.base_tomo_name + '-bin4-wbp.rec')
        self.ori_tomo_path = None
        # self.ori_tomo_path = os.path.join(self.current_path, tomo_name, self.base_tomo_name + '-bin4-5i.rec')
        self.rec_tomo_path = os.path.join(self.current_path, tomo_name, 'ves_seg', 'tomoset', self.base_tomo_name + '_wbp_resample.mrc')
        self.tomograms_star_path = os.path.join(self.current_path, tomo_name, 'ves_seg', 'tomograms.star')
        self.deconv_tomo_path = os.path.join(self.current_path, tomo_name, 'ves_seg', 'tomo_deconv', self.base_tomo_name + '_wbp_resample.mrc')
        self.deconv_para = os.path.join(self.current_path, tomo_name, 'ves_seg', 'tomo_deconv', self.base_tomo_name + '.json')
        self.correction_output_path = os.path.join(self.current_path, tomo_name, 'ves_seg')
        self.area_path = os.path.join(self.current_path, tomo_name, 'area.mod')
        self.area_by_sam_path = os.path.join(self.current_path, tomo_name, 'sam_area.mrc')
        self.memb_prompt_path = os.path.join(self.current_path, tomo_name, 'ves_seg', 'membrane', 'prompt.mod')
        self.memb_folder_path = os.path.join(self.current_path, tomo_name, 'ves_seg', 'membrane')
        self.memb_result_path = os.path.join(self.current_path, tomo_name, 'ves_seg', 'membrane', self.base_tomo_name + '.mod')
        self.memb_manual_path = os.path.join(self.current_path, tomo_name, 'ves_seg', 'membrane', 'premembrane.mod')
        self.new_memb_result_path = os.path.join(self.current_path, tomo_name, 'ves_seg', 'membrane', self.base_tomo_name + '_new.mod')
        self.manualy_memb_path = os.path.join(self.current_path, tomo_name, 'ves_seg', 'membrane', self.base_tomo_name + '_manual.mod')
        
        
        self.isonet_tomo_path = os.path.join(self.current_path, tomo_name, 'ves_seg', self.base_tomo_name + '_wbp_corrected.mrc')
        # temp
        # self.isonet_tomo_path = os.path.join(self.current_path, tomo_name, self.base_tomo_name + '_wbp_corrected.mrc')
        self.ori_xml_path = os.path.join(self.current_path, tomo_name, 'ves_seg', 'vesicle_analysis', self.base_tomo_name + '_ori.xml')
        self.ori_filter_xml_path = os.path.join(self.current_path, tomo_name, 'ves_seg', 'vesicle_analysis', self.base_tomo_name + '_filter_bak.xml')
        self.filter_xml_path = os.path.join(self.current_path, tomo_name, 'ves_seg', 'vesicle_analysis', self.base_tomo_name + '_filter.xml')
        self.ori_class_xml_path = os.path.join(self.current_path, tomo_name, 'ves_seg', 'vesicle_analysis', self.base_tomo_name + '_vesicle_class_bak.xml')
        self.class_xml_path = os.path.join(self.current_path, tomo_name, 'ves_seg', 'vesicle_analysis', self.base_tomo_name + '_vesicle_class.xml')
        self.final_xml_path = os.path.join(self.current_path, tomo_name, 'ves_seg', 'vesicle_analysis', self.base_tomo_name + '.xml')
        self.tether_xml_path = os.path.join(self.current_path, tomo_name, 'ves_seg', 'vesicle_analysis', self.base_tomo_name + '_tether.xml')
        self.contact_xml_path = os.path.join(self.current_path, tomo_name, 'ves_seg', 'vesicle_analysis', self.base_tomo_name + '_contact.xml')
        self.omega_xml_path = os.path.join(self.current_path, tomo_name, 'ves_seg', 'vesicle_analysis', self.base_tomo_name + '_omega.xml')
        self.weidong_excel_path = os.path.join(self.current_path, tomo_name, 'ves_seg', 'vesicle_analysis', self.base_tomo_name + '_wd.xlsx')
        self.segment_path = os.path.join(self.current_path, tomo_name, 'ves_seg', self.base_tomo_name + '_segment.mrc')
        
        self.label_path = os.path.join(self.current_path, tomo_name, 'ves_seg', self.base_tomo_name + '_label_vesicle.mrc')
        self.json_file_path = os.path.join(self.current_path, tomo_name, 'ves_seg', self.base_tomo_name + '_vesicle.json')
        # temp
        # self.label_path = os.path.join(self.current_path, tomo_name, self.base_tomo_name + '_label_vesicle.mrc')
        # self.json_file_path = os.path.join(self.current_path, tomo_name, self.base_tomo_name + '_vesicle.json')
        self.xlsx_file_path = os.path.join(self.current_path, tomo_name, 'ves_seg', self.base_tomo_name + '_vesicle.xlsx')
        self.new_json_file_path = os.path.join(self.root_dir, 'vesicle_new_{}.json'.format(self.pid))
        self.new_label_file_path = os.path.join(self.root_dir, 'label_{}.mrc'.format(self.pid))
        self.point_file_path = os.path.join(self.root_dir, 'points_{}.point'.format(self.pid))
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
        # elif os.path.exists(self.ori_tomo_path):
        #     return self.ProgressStage.MAKE_DECONVOLUTION
        else:
            return self.ProgressStage.OPEN_TOMO
        
    def update_progress_stage(self):
        """更新进度阶段"""
        self.progress_stage = self.determine_progress()

    @property
    def process_temp_paths(self):
        return [path for path in [
            self.new_json_file_path,
            self.new_label_file_path,
            self.point_file_path,
        ] if path]

    def _audit_log_path(self):
        root_dir = self.root_dir
        if not root_dir:
            root_dir = os.path.join(os.path.expanduser("~"), ".segvesicle")
        return os.path.join(root_dir, "cleanup_audit.log")

    def _stack_summary(self, limit=12):
        stack = traceback.extract_stack(limit=limit)
        # drop helper frames: _stack_summary and record_audit_event
        stack = stack[:-2]
        return [
            {
                "file": frame.filename,
                "line": frame.lineno,
                "function": frame.name,
            }
            for frame in stack
        ]

    def record_audit_event(self, source, event, details=None):
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pid": self.pid,
            "source": source,
            "event": event,
            "details": details or {},
            "stack": self._stack_summary(),
        }

        log_path = self._audit_log_path()
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError:
            # Never block core workflow if audit logging fails.
            pass

    def cleanup_process_temp_files(self, source="unknown"):
        cleanup_result = {
            'removed': [],
            'missing': [],
            'failed': [],
        }

        for temp_path in self.process_temp_paths:
            if not os.path.exists(temp_path):
                cleanup_result['missing'].append(temp_path)
                continue

            try:
                os.remove(temp_path)
                cleanup_result['removed'].append(temp_path)
            except OSError as error:
                cleanup_result['failed'].append((temp_path, str(error)))

        self.record_audit_event(
            source=source,
            event="cleanup_process_temp_files",
            details=cleanup_result,
        )
        return cleanup_result
