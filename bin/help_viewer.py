import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QListWidget, QTextBrowser
from PyQt5.QtCore import QUrl  # 导入 QUrl
import markdown

class HelpViewer(QMainWindow):
    def __init__(self):
        super().__init__()
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
        markdown_dir = os.path.join(script_dir, 'help_files')
        if os.path.exists(markdown_dir):
            for file_name in os.listdir(markdown_dir):
                if file_name.endswith('.md'):
                    self.md_files.append(os.path.join(markdown_dir, file_name))
                    self.file_list.addItem(file_name)

    def display_help_content(self, item):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'help_files', item.text())
        if file_path in self.md_files:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_content = file.read()
                html_content = markdown.markdown(md_content)
                base_url = QUrl.fromLocalFile(os.path.dirname(file_path))
                base_url_path = base_url.toString()  # 获取 Base URL
                # 调整路径
                html_content = html_content.replace('src="img/', f'src="{base_url_path}/img/')
                # 添加 CSS 样式以确保代码块换行
                
                self.help_content.setHtml(html_content)
                # print(html_content)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = HelpViewer()
    viewer.show()
    sys.exit(app.exec_())
