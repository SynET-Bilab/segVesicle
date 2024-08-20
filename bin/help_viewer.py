import sys
import os
import base64
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QListWidget
from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
import markdown
import re

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
        
        self.help_content = QWebEngineView()  # 使用 QWebEngineView
        
        self.layout.addWidget(self.file_list)
        self.layout.addWidget(self.help_content)

        self.load_md_files()
        # 自动加载并显示 GUI.md
        self.open_default_file("3. Gui.md")

    def load_md_files(self):
        self.md_files = []
        script_dir = os.path.dirname(os.path.abspath(__file__))
        markdown_dir = os.path.join(script_dir, 'help_files')
        if os.path.exists(markdown_dir):
            # 获取所有 Markdown 文件并按名称排序
            md_files = sorted([file_name for file_name in os.listdir(markdown_dir) if file_name.endswith('.md')])
            for file_name in md_files:
                file_path = os.path.join(markdown_dir, file_name)
                self.md_files.append(file_path)
                self.file_list.addItem(file_name)

    def convert_image_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string

    def display_help_content(self, item):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'help_files', item.text())
        if file_path in self.md_files:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_content = file.read()
                html_content = markdown.markdown(md_content)
                base_url = QUrl.fromLocalFile(os.path.dirname(file_path))
                base_url_path = base_url.toString()

                # 查找并处理所有图片路径
                img_pattern = re.compile(r'src="img/([^"]+)"')
                matches = img_pattern.findall(html_content)
                for image_relative_path in matches:
                    image_absolute_path = os.path.join(script_dir, 'help_files', 'img', image_relative_path)
                    if os.path.exists(image_absolute_path):
                        base64_image = self.convert_image_to_base64(image_absolute_path)
                        html_content = html_content.replace(f'src="img/{image_relative_path}"', f'src="data:image/png;base64,{base64_image}"')

                # 添加 CSS 样式以确保代码块换行
                html_content = """
                <style>
                    code, pre {
                        white-space: pre-wrap;
                        word-wrap: break-word;
                    }
                </style>
                """ + html_content

                self.help_content.setHtml(html_content)

    def open_default_file(self, default_file_name):
        # 检查 GUI.md 是否在文件列表中
        items = [self.file_list.item(i) for i in range(self.file_list.count())]
        for item in items:
            if item.text() == default_file_name:
                self.file_list.setCurrentItem(item)
                self.display_help_content(item)
                break

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = HelpViewer()
    viewer.show()
    sys.exit(app.exec_())
