from qtpy.QtWidgets import QMainWindow, QVBoxLayout, QLabel, QLineEdit, QPushButton, QWidget, QFormLayout, QProgressDialog, QDesktopWidget
from qtpy.QtCore import Qt
import subprocess

class MembSegmentationWindow(QMainWindow):
    def __init__(self, tomo_viewer):
        # 调用父类初始化，传入 qt_viewer 窗口
        super().__init__(tomo_viewer.viewer.window.qt_viewer)

        self.tomo_viewer = tomo_viewer  # 保存传入的 tomo_viewer 对象

        # 设置窗口标题
        self.setWindowTitle("Membrane Segmentation Parameters")

        # 调整窗口大小，增加宽度
        self.resize(500, 100)

        # 主部件和布局
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # 表单布局，用于输入 pixel_size 和 extend
        self.form_layout = QFormLayout()

        # Pixel size 输入
        self.pixel_size_input = QLineEdit(self)
        self.pixel_size_input.setPlaceholderText("Enter pixel size (default: Read from the header of tomo)")
        self.form_layout.addRow(QLabel("Pixel Size:"), self.pixel_size_input)

        # Extend 输入，默认值为 30
        self.extend_input = QLineEdit(self)
        self.extend_input.setPlaceholderText("Enter extend value (default: 30)")
        self.extend_input.setText("30")
        self.form_layout.addRow(QLabel("Extend:"), self.extend_input)

        # 将表单布局添加到主布局
        self.layout.addLayout(self.form_layout)

        # Apply 按钮
        self.apply_button = QPushButton("Apply", self)
        self.apply_button.clicked.connect(self.apply_segmentation)
        self.layout.addWidget(self.apply_button)
        
        # 调用方法使窗口居中
        self.center_on_screen()

    def apply_segmentation(self):
        # 获取用户输入的 pixel_size 和 extend
        pixel_size = self.pixel_size_input.text()
        pixel_size = float(pixel_size) if pixel_size else None  # 转换为 float 或保持 None

        extend = self.extend_input.text()
        extend = int(extend) if extend else 30  # 默认值为 30

        # 执行膜分割
        self.register_seg_memb(pixel_size=pixel_size, extend=extend)

    def register_seg_memb(self, pixel_size=None, extend=30):
        # 显示进度对话框
        progress_dialog = QProgressDialog("Processing...", 'Cancel', 0, 100, self)
        progress_dialog.setWindowTitle('Opening')
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setValue(0)
        progress_dialog.show()

        # 设置输出路径
        output_path = self.tomo_viewer.tomo_path_and_stage.memb_folder_path + '/' + self.tomo_viewer.tomo_path_and_stage.base_tomo_name
        cmd = f'segprepost.py run {self.tomo_viewer.tomo_path_and_stage.isonet_tomo_path} {self.tomo_viewer.tomo_path_and_stage.memb_prompt_path} -o {output_path}'

        # 如果 Pixel size 不为 None，则添加 --pixel 参数
        if pixel_size is not None:
            cmd += f' --pixel {pixel_size}'

        # 添加 --extend 参数，默认值为 30
        cmd += f' --extend {extend}'

        try:
            # 运行命令并捕获错误
            subprocess.run(cmd, shell=True, check=True)

            # 成功时的提示信息
            result = f'Membrane segmentation successful. The result is saved at {self.tomo_viewer.tomo_path_and_stage.memb_folder_path}. You can click Visualize to view the result.'
            self.tomo_viewer.print(result)

        except subprocess.CalledProcessError as e:
            # 捕获错误并输出错误信息
            error_message = f"An error occurred during membrane segmentation: {str(e)}"
            self.tomo_viewer.print(error_message)

        # 进度完成
        progress_dialog.setValue(100)
        
    def center_on_screen(self):
        ''' 使窗口居中显示 '''
        qr = self.frameGeometry()  # 获取窗口的矩形几何尺寸
        cp = QDesktopWidget().availableGeometry().center()  # 获取屏幕的中心位置
        qr.moveCenter(cp)  # 将窗口的矩形几何中心移动到屏幕中心
        self.move(qr.topLeft())  # 将窗口移动到新位置