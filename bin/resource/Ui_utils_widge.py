from PyQt5 import QtWidgets, QtCore

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        
        # 使用QVBoxLayout管理窗口布局
        self.mainLayout = QtWidgets.QVBoxLayout(Form)
        
        # 创建TabWidget，并添加到布局中
        self.tabWidget = QtWidgets.QTabWidget(Form)
        self.tabWidget.setObjectName("tabWidget")
        self.mainLayout.addWidget(self.tabWidget)
        
        # GuideLine Tab
        self.GuideLine = QtWidgets.QWidget()
        self.GuideLine.setObjectName("GuideLine")
        
        # 在GuideLine中设置布局
        self.guideLineLayout = QtWidgets.QVBoxLayout(self.GuideLine)
        
        # 顶部的布局（标签和按钮）
        self.topLayout = QtWidgets.QHBoxLayout()
        self.label = QtWidgets.QLabel(self.GuideLine)
        self.label.setObjectName("label")
        self.topLayout.addWidget(self.label)
        
        self.help = QtWidgets.QPushButton(self.GuideLine)
        self.help.setObjectName("help")
        self.topLayout.addWidget(self.help)
        
        self.guideLineLayout.addLayout(self.topLayout)
        
        # ProgressStage
        self.progressStage = QtWidgets.QLineEdit(self.GuideLine)
        self.progressStage.setEnabled(True)
        self.progressStage.setMouseTracking(True)
        self.progressStage.setObjectName("progressStage")
        self.guideLineLayout.addWidget(self.progressStage)
        
        # 标签和TextEdit
        self.label_2 = QtWidgets.QLabel(self.GuideLine)
        self.label_2.setObjectName("label_2")
        self.guideLineLayout.addWidget(self.label_2)
        
        self.terminal = QtWidgets.QTextEdit(self.GuideLine)
        self.terminal.setEnabled(True)
        self.terminal.setMouseTracking(True)
        self.terminal.setReadOnly(True)
        self.terminal.setObjectName("terminal")
        self.guideLineLayout.addWidget(self.terminal)
        
        self.tabWidget.addTab(self.GuideLine, "")
        
        # Summary Tab
        self.Summary = QtWidgets.QWidget()
        self.Summary.setObjectName("Summary")
        self.tabWidget.addTab(self.Summary, "")

        self.retranslateUi(Form)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "Progress Stage:"))
        self.help.setText(_translate("Form", "?"))
        self.label_2.setText(_translate("Form", "Terminal"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.GuideLine), _translate("Form", "GuideLine"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Summary), _translate("Form", "Summary"))
