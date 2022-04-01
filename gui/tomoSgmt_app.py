'''

'''
import sys,os

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QTableWidgetItem,QMessageBox,QDialog
from PyQt5.QtCore import QObject, pyqtSlot, QProcess
from tomoSgmt.gui.tomoSgmt_gui import Ui_MainWindow


class Model:
    def __init__(self):
        self.pwd = os.getcwd().replace("\\", "/")
        self.log_file = "log.txt"
        self.btn_pressed_text = None

    def isValid(self, fileName):
        '''
        returns True if the file exists and can be
        opened.  Returns False otherwise.
        '''
        try:
            file = open(fileName, 'r')
            file.close()
            return True
        except:
            return False

    def isValidPath(self, path):
        '''
        returns True if the file exists and can be
        opened.  Returns False otherwise.
        '''
        try:
            isDir = os.path.isdir(path)
            return isDir
        except:
            return False

    def is_number(self, string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    def is_file_exist(self, path, suffix):
        fileList = []
        for fname in os.listdir(path):
            if fname.endswith(suffix):
                # do stuff on the file
                fileList.append(fname)
        return fileList

    def sim_path(self, pwd, path):
        if pwd in path:
            return "." + path[len(pwd):]
        else:
            return path

    def getLogContent(self, fileName):
        '''
        sets the member fileName to the value of the argument
        if the file exists.  Otherwise resets both the filename
        and file contents members.
        '''
        if self.isValid(fileName):
            self.fileName = fileName
            content = open(fileName, 'r').read()
            return content
        else:
            return None


class MainWindowUIClass(Ui_MainWindow):
    def __init__(self, mainwindow):
        '''Initialize the super class
        '''
        super().__init__()
        self.setupUi(mainwindow)
        self.initUi()

        self.model = Model()
        # reset process as None
        self.p = None
        self.previous_log_line = ""


    def initUi(self):
        self.toolButton_file_1.clicked.connect(lambda: self.browseSlot("tomo_file"))
        self.toolButton_file_2.clicked.connect(lambda: self.browseSlot("trained_model"))
        self.toolButton_file_4.clicked.connect(lambda: self.browseSlot("area_file"))
        self.toolButton_file_3.clicked.connect(lambda: self.browseFolderSlot("result_folder"))
        self.pushButton_add_models.clicked.connect(lambda: self.browsesSlot("add_models"))

        self.pushButton_3dmod_view.clicked.connect(self.view_3dmod)
        self.pushButton_3dmod_view_2.clicked.connect(self.view_3dmod)
        #self.pushButton_add_models.clicked.connect(self.add_models)
        self.pushButton_resample.clicked.connect(self.resample)
        self.pushButton_predict.clicked.connect(self.predict)
        self.pushButton_morph_process.clicked.connect(self.morph_process)
        self.pushButton_view_result_with_tomogram.clicked.connect(self.view_result)

        ves_seg_path = os.popen("which ves_seg.py").read()
        tmp = ves_seg_path.split("bin/ves_seg.py")
        root_path = tmp[0]
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(root_path+"gui/icons/icon_folder.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_file_1.setIcon(icon)
        self.toolButton_file_2.setIcon(icon)
        self.toolButton_file_3.setIcon(icon)
        self.toolButton_file_4.setIcon(icon)

        ###Set up log file monitor###
        import datetime
        now = datetime.datetime.now()

        # create an empty log file
        self.model = Model()
        if not self.model.isValid(self.model.log_file):
            os.system("echo {} > {}".format(now.strftime("%Y-%m-%d %H:%M:%S"), self.model.log_file))

        self.textBrowser_log.setText(self.model.getLogContent(self.model.log_file))
        self.textBrowser_log.moveCursor(QtGui.QTextCursor.End)


    def start_process(self, cmd, btn):
        self.p = None
        if self.p is None:  # No process running.
            self.p = QProcess()
            #change the status of the current botton
            if btn.text() in ["Deconvolve","Generate Mask","Extract","Refine","Predict"]:
                self.model.btn_pressed_text =  btn.text()
                btn.setText("Stop")
                btn.setStyleSheet('QPushButton {color: red;}')
            else:
                btn.setEnabled(False)
            self.p.readyReadStandardOutput.connect(self.dataReady)
            self.p.finished.connect(lambda: self.process_finished(btn))
            self.p.start(cmd)

        elif btn.text() =="Stop":
            if self.p:
                self.p.kill()
            else:
                if self.model.btn_pressed_text:
                    btn.setText(self.model.btn_pressed_text)
        else:
            self.warn_window("Already runing another job, please wait until it finished!")

    def dataReady(self):
        cursor = self.textBrowser_log.textCursor()
        cursor.movePosition(cursor.End)
        # have transfer byte string to unicode string
        import string
        printable = set(string.printable)
        printable.add(u'\u2588')

        txt = str(self.p.readAll(), 'utf-8')
        # txt += self.mw.p.errorString()

        printable_txt = "".join(list(filter(lambda x: x in printable, txt)))

        if '[' in self.previous_log_line and '[' in printable_txt:
            cursor.movePosition(cursor.StartOfLine, cursor.MoveAnchor)
            cursor.movePosition(cursor.End, cursor.KeepAnchor)
            cursor.removeSelectedText()
            cursor.deletePreviousChar()
        cursor.insertText(printable_txt)
        f = open(self.model.log_file, 'a+')
        f.write(printable_txt)
        f.close()

        self.previous_log_line = printable_txt
        self.textBrowser_log.ensureCursorVisible()
        self.textBrowser_log.moveCursor(QtGui.QTextCursor.End)


    def process_finished(self, btn):
        if btn.text() == "Stop":
            if self.model.btn_pressed_text:
                btn.setText(self.model.btn_pressed_text)
                #btn.setText("Refine")
                self.model.btn_pressed_text = None
                btn.setStyleSheet('QPushButton {color: black;}')
        else:
            btn.setEnabled(True)
        self.mw.p = None


    def switch_btn(self, btn):
        switcher = {
            "tomo_file": self.lineEdit_tomogram_file_name,
            "trained_model": self.lineEdit_Trained_model,
            "area_file": self.lineEdit_area_file,
            "result_folder": self.lineEdit_result_folder,
            "add_models": self.lineEdit_Trained_model
        }
        return switcher.get(btn, "Invaid btn name")


    def browseSlot(self, btn):
        '''
        Called when the user presses the Browse button
        '''
        lineEdit = self.switch_btn(btn)
        pwd = os.getcwd().replace("\\", "/")
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog

        flt = "All Files (*)"
        # if btn == "tomo_file":
        #     flt = "rec file (*.rec);;All Files (*)"
        if btn == "trained_model":
            flt = "model file (*.h5);;All Files (*)"
        if btn == "area_file":
            flt = "area file (*.point);;All Files (*)"

        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Choose File",
            "",
            flt,
            options=options)
        if fileName:
            simple_name = self.model.sim_path(pwd, fileName)
            lineEdit.setText(simple_name)


    def browsesSlot(self, btn):
        '''

        '''
        lineEdit = self.switch_btn(btn)
        pwd = os.getcwd().replace("\\", "/")
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog

        flt = "All Files (*)"
        if btn == "add_models":
            flt = "model file (*.h5);;All Files (*)"
        fileNames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            None,
            "Choose File",
            "",
            flt,
            options=options
        )
        if lineEdit.text():
            record = lineEdit.text()
        else:
            record = ''
        if fileNames:
            simple_name = []
            simple_name.append(record)
            for fileName in fileNames:
                simple_name.append(self.model.sim_path(pwd, fileName))
            text = ','.join(simple_name)
            lineEdit.setText(text)


    def browseFolderSlot(self, btn):
        '''
            Called when the user presses the Browse folder button
        '''
        lineEdit = self.switch_btn(btn)
        try:
            pwd = os.getcwd().replace("\\","/")
            dir_path=QtWidgets.QFileDialog.getExistingDirectory(None,"Choose Directory",pwd)
            simple_path = self.model.sim_path(pwd,dir_path)
            lineEdit.setText( simple_path )
        except:
            pass


    def resample(self):
        cmd = 'ves_seg.py resample '
        if self.lineEdit_tomogram_file_name.text():
            cmd = '{} --tomo {}'.format(cmd, self.lineEdit_tomogram_file_name.text())
        if self.lineEdit_pixel_size.text():
            cmd = '{} --pixel_size {}'.format(cmd, self.lineEdit_pixel_size.text())

        self.start_process(cmd, self.pushButton_resample)


    def predict(self):
        cmd = 'ves_seg.py predict '
        if self.lineEdit_Trained_model.text():
            cmd = '{} --model {}'.format(cmd, self.lineEdit_Trained_model.text())
        if self.lineEdit_result_folder.text():
            cmd = '{} --dir {}'.format(cmd, self.lineEdit_result_folder.text())
        if self.lineEdit_GPU_ID.text():
            cmd = '{} --gpuID {}'.format(cmd, self.lineEdit_GPU_ID.text())
        if self.checkBox_use_resampled_tomo.isChecked():
            tomo = self.lineEdit_tomogram_file_name.text()
            cmd = '{} --mrc {}'.format(cmd, tomo.split('.')[0] + '-resample.' + tomo.split('.')[1])
        else:
            if self.lineEdit_tomogram_file_name.text():
                cmd = '{} --mrc {}'.format(cmd, self.lineEdit_tomogram_file_name.text())

        self.start_process(cmd, self.pushButton_predict)


    def morph_process(self):
        cmd = 'ves_seg.py morph '
        if self.lineEdit_result_folder.text():
            cmd = '{} --dir {}'.format(cmd, self.lineEdit_result_folder.text())
        if self.lineEdit_area_file.text():
            cmd = '{} --area_file {}'.format(cmd, self.lineEdit_area_file.text())
        if self.lineEdit_tomogram_file_name.text():
            tomo = self.lineEdit_tomogram_file_name.text()
            root_name = tomo.split('/')[-1].split('.')[0]
            mask_file = self.lineEdit_result_folder.text() + '/' + root_name + '-mask.mrc'
            cmd = '{} --mask_file {}'.format(cmd, mask_file)

        self.start_process(cmd, self.pushButton_morph_process)


    def json2mod(self, json_file):

        with open(json_file, "r") as t:
            data = eval(t.read()).get('vesicles')
        center = []
        for i in range(len(data)):
            center.append(data[i].get('center'))
        for i in range(len(center)):
            center[i][0], center[i][1], center[i][2] = center[i][2], center[i][1], center[i][0]

        with open("point.txt", "w") as p:
            contour = 0
            for i in center:
                contour = contour + 1
                p.write('{} '.format(contour))
                for j in i:
                    p.write(str(j))
                    p.write(' ')
                p.write('\n')

        cmd = 'point2model point.txt point.mod'
        with open("point.txt", "r") as p:
            os.system(cmd)


    def view_3dmod(self):
        cmd = '3dmod'
        if self.lineEdit_tomogram_file_name.text():
            item_text = self.lineEdit_tomogram_file_name.text()
            if item_text[-4:] == '.mrc' or item_text[-4:] == '.rec':
                cmd = '{} {}'.format(cmd, item_text)
                os.system(cmd)
            else:
                self.warn_window("selected items are not mrc or rec file(s)")


    def view_result(self):
        cmd = '3dmod {} {}'.format(self.lineEdit_tomogram_file_name.text(), 'point.mod')
        if self.lineEdit_tomogram_file_name.text():
            item_text = self.lineEdit_tomogram_file_name.text()
            if item_text[-4:] == '.mrc' or item_text[-4:] == '.rec':
                root_name = item_text.split('/')[-1].split('-')[0]
                dir = self.lineEdit_result_folder.text()
                output_file_in_area = dir + '/' + root_name + '_vesicle_in.json'
                self.json2mod(output_file_in_area)
                os.system(cmd)
            else:
                self.warn_window("selected items are not mrc or rec file(s)")


    def warn_window(self,text):
        msg = QMessageBox()
        msg.setWindowTitle("Warning!")
        msg.setText(text)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setIcon(QMessageBox.Warning)
        msg.exec_()

class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.p = None

    def closeEvent(self, event):
        if self.p:

            result = QtWidgets.QMessageBox.question(self,
                                                    "Confirm Exit...",
                                                    "Do you want to continue the existing job in the background?",
                                                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel)
            event.ignore()
            if result == QtWidgets.QMessageBox.Yes:
                event.accept()
            if result == QtWidgets.QMessageBox.No:
                self.p.kill()
                event.accept()
                # kill the old process
        else:
            result = QtWidgets.QMessageBox.question(self,
                                                    "Confirm Exit...",
                                                    "Do you want to exit? ",
                                                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            event.ignore()
            if result == QtWidgets.QMessageBox.Yes:
                event.accept()
            if result == QtWidgets.QMessageBox.No:
                pass
                # kill the old process


def main():
    app = QtWidgets.QApplication(sys.argv)
    mainwindow = QtWidgets.QMainWindow()
    MainWindow = MainWindowUIClass(mainwindow)
    #MainWindow = QtWidgets.QMainWindow()
    # ui = MainWindowUIClass()
    # ui.setupUi(MainWindow)
    mainwindow.show()
    sys.exit(app.exec_())

main()










