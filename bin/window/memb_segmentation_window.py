from qtpy.QtWidgets import (
    QMainWindow, QVBoxLayout, QLabel, QLineEdit, QPushButton, QWidget,
    QFormLayout, QDesktopWidget, QComboBox
)
from qtpy.QtCore import Qt, QObject, QThread, Signal
import subprocess

class Worker(QObject):
    finished = Signal()
    error = Signal(str)

    def __init__(self, cmd):
        super().__init__()
        self.cmd = cmd
        self.process = None

    def run(self):
        try:
            # 启动子进程
            self.process = subprocess.Popen(self.cmd, shell=True)
            # 等待子进程结束
            self.process.wait()
            if self.process.returncode == 0:
                self.finished.emit()
            else:
                self.error.emit(f"Process exited with code {self.process.returncode}")
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()

class MembSegmentationWindow(QMainWindow):
    def __init__(self, tomo_viewer):
        super().__init__(tomo_viewer.viewer.window.qt_viewer)

        self.tomo_viewer = tomo_viewer
        self.worker_thread = None
        self.worker = None

        self.setWindowTitle("Membrane Segmentation Parameters")
        self.resize(500, 150)

        # 主布局
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # 表单布局
        self.form_layout = QFormLayout()

        # Pixel size 输入
        self.pixel_size_input = QLineEdit(self)
        self.pixel_size_input.setPlaceholderText("Enter pixel size (default: Read from the header of tomo)")
        self.form_layout.addRow(QLabel("Pixel Size:"), self.pixel_size_input)

        # Extend 输入
        self.extend_input = QLineEdit(self)
        self.extend_input.setPlaceholderText("Enter extend value (default: 30)")
        self.extend_input.setText("30")
        self.form_layout.addRow(QLabel("Extend:"), self.extend_input)

        # 选项框
        self.method_selection = QComboBox(self)
        self.method_selection.addItems(["segprepost", "segonemem"])
        self.form_layout.addRow(QLabel("Segmentation Method:"), self.method_selection)

        self.layout.addLayout(self.form_layout)

        # Apply 按钮
        self.apply_button = QPushButton("Apply", self)
        self.apply_button.clicked.connect(self.apply_segmentation)
        self.layout.addWidget(self.apply_button)

        self.center_on_screen()

    def apply_segmentation(self):
        pixel_size = self.pixel_size_input.text()
        pixel_size = float(pixel_size) if pixel_size else None

        extend = self.extend_input.text()
        extend = int(extend) if extend else 30

        selected_method = self.method_selection.currentText()
        output_path = f"{self.tomo_viewer.tomo_path_and_stage.memb_folder_path}/{self.tomo_viewer.tomo_path_and_stage.base_tomo_name}"
        
        cmd = f'{selected_method}.py run {self.tomo_viewer.tomo_path_and_stage.isonet_tomo_path} {self.tomo_viewer.tomo_path_and_stage.memb_prompt_path} -o {output_path}'
        
        if pixel_size is not None:
            cmd += f' --pixel {pixel_size}'
        cmd += f' --extend {extend}'

        # 创建后台线程并启动
        self.worker_thread = QThread()
        self.worker = Worker(cmd)
        self.worker.moveToThread(self.worker_thread)

        # 连接信号
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_segmentation_finished)
        self.worker.error.connect(self.on_segmentation_error)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        # 启动线程
        self.worker_thread.start()

        # 禁用 Apply 按钮
        self.apply_button.setEnabled(False)

    def on_segmentation_finished(self):
        result = f'Membrane segmentation successful. The result is saved at {self.tomo_viewer.tomo_path_and_stage.memb_folder_path}. You can click Visualize to view the result.'
        self.tomo_viewer.print(result)
        self.apply_button.setEnabled(True)
        
        # Automatically close the window upon successful completion
        self.close()

    def on_segmentation_error(self, error_message):
        self.tomo_viewer.print(f"An error occurred during membrane segmentation: {error_message}")
        self.apply_button.setEnabled(True)

    def closeEvent(self, event):
        # 不停止线程，允许后台继续运行
        if self.worker and self.worker_thread.isRunning():
            self.worker_thread.finished.disconnect(self.worker_thread.deleteLater)
            self.worker_thread.quit()
        event.accept()

    def center_on_screen(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())