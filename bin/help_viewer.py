import sys
import os
import markdown
import napari
from qtpy.QtWidgets import QApplication, QMainWindow, QTextBrowser, QListWidget, QHBoxLayout, QWidget
from qtpy.QtCore import Qt, QUrl
from pygments.formatters import HtmlFormatter
from markdown.extensions.codehilite import CodeHiliteExtension

class HelpViewer(QMainWindow):
    def __init__(self, qt_viewer):
        super().__init__(qt_viewer)
        # self.qt_viewer = qt_viewer
        
        self.setWindowTitle("Help Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QHBoxLayout(self.main_widget)

        self.file_list = QListWidget()
        self.file_list.setMaximumWidth(200)
        self.file_list.itemClicked.connect(self.display_help_content)
        
        self.help_content = QTextBrowser()
        self.help_content.setOpenExternalLinks(True)
        
        self.layout.addWidget(self.file_list)
        self.layout.addWidget(self.help_content)

        self.load_md_files()

    def load_md_files(self):
        self.md_files = []
        script_dir = os.path.dirname(os.path.abspath(__file__))
        markdown_file = os.path.join(script_dir, 'help_files')
        # help_dir = "./help_files"  # 替换为你的帮助文件目录路径
        if os.path.exists(markdown_file):
            for file_name in os.listdir(markdown_file):
                if file_name.endswith('.md'):
                    self.md_files.append(os.path.join(markdown_file, file_name))
                    self.file_list.addItem(file_name)

    def display_help_content(self, item):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'help_files', item.text())
        if file_path in self.md_files:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_content = file.read()
                self.help_content.setMarkdown(md_content)
                
                
if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = HelpViewer()
    viewer.show()
    sys.exit(app.exec_())