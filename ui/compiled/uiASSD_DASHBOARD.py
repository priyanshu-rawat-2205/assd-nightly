# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ASSD_DASHBOARD.ui'
##
## Created by: Qt User Interface Compiler version 6.6.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QPushButton, QScrollArea,
    QSizePolicy, QSlider, QTabWidget, QTextEdit,
    QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1366, 640)
        MainWindow.setStyleSheet(u"")
        self.actionSettings = QAction(MainWindow)
        self.actionSettings.setObjectName(u"actionSettings")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.sideFrame = QFrame(self.centralwidget)
        self.sideFrame.setObjectName(u"sideFrame")
        self.sideFrame.setGeometry(QRect(0, 0, 231, 631))
        self.sideFrame.setFrameShape(QFrame.StyledPanel)
        self.sideFrame.setFrameShadow(QFrame.Raised)
        self.settingGroup = QGroupBox(self.sideFrame)
        self.settingGroup.setObjectName(u"settingGroup")
        self.settingGroup.setGeometry(QRect(0, 10, 231, 461))
        self.scrollArea = QScrollArea(self.settingGroup)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setGeometry(QRect(2, 22, 231, 431))
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents_2 = QWidget()
        self.scrollAreaWidgetContents_2.setObjectName(u"scrollAreaWidgetContents_2")
        self.scrollAreaWidgetContents_2.setGeometry(QRect(0, 0, 215, 828))
        self.gridLayout_3 = QGridLayout(self.scrollAreaWidgetContents_2)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.scrollAreaFrame = QFrame(self.scrollAreaWidgetContents_2)
        self.scrollAreaFrame.setObjectName(u"scrollAreaFrame")
        self.scrollAreaFrame.setMinimumSize(QSize(190, 810))
        self.scrollAreaFrame.setMaximumSize(QSize(300, 810))
        self.scrollAreaFrame.setFrameShape(QFrame.StyledPanel)
        self.scrollAreaFrame.setFrameShadow(QFrame.Raised)
        self.camGroup = QGroupBox(self.scrollAreaFrame)
        self.camGroup.setObjectName(u"camGroup")
        self.camGroup.setGeometry(QRect(0, 0, 201, 123))
        self.camGroup.setMinimumSize(QSize(201, 111))
        self.gridLayout_4 = QGridLayout(self.camGroup)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.camDeviceComboBox = QComboBox(self.camGroup)
        self.camDeviceComboBox.addItem("")
        self.camDeviceComboBox.setObjectName(u"camDeviceComboBox")

        self.gridLayout_4.addWidget(self.camDeviceComboBox, 1, 0, 1, 2)

        self.startButton = QPushButton(self.camGroup)
        self.startButton.setObjectName(u"startButton")

        self.gridLayout_4.addWidget(self.startButton, 2, 0, 1, 1)

        self.stopButton = QPushButton(self.camGroup)
        self.stopButton.setObjectName(u"stopButton")

        self.gridLayout_4.addWidget(self.stopButton, 2, 1, 1, 1)

        self.deviceLabel = QLabel(self.camGroup)
        self.deviceLabel.setObjectName(u"deviceLabel")

        self.gridLayout_4.addWidget(self.deviceLabel, 0, 0, 1, 2)

        self.maskGroup = QGroupBox(self.scrollAreaFrame)
        self.maskGroup.setObjectName(u"maskGroup")
        self.maskGroup.setGeometry(QRect(0, 120, 201, 311))
        self.maskGroup.setMinimumSize(QSize(201, 281))
        self.getMaskCheckBox = QCheckBox(self.maskGroup)
        self.getMaskCheckBox.setObjectName(u"getMaskCheckBox")
        self.getMaskCheckBox.setGeometry(QRect(10, 30, 171, 31))
        self.colorComboBox = QComboBox(self.maskGroup)
        self.colorComboBox.addItem("")
        self.colorComboBox.setObjectName(u"colorComboBox")
        self.colorComboBox.setGeometry(QRect(10, 60, 181, 25))
        self.colorComboBox.setEditable(False)
        self.colorButton = QPushButton(self.maskGroup)
        self.colorButton.setObjectName(u"colorButton")
        self.colorButton.setGeometry(QRect(10, 100, 181, 25))
        self.targetGroup = QGroupBox(self.scrollAreaFrame)
        self.targetGroup.setObjectName(u"targetGroup")
        self.targetGroup.setGeometry(QRect(0, 430, 201, 381))
        self.targetGroup.setMinimumSize(QSize(201, 281))
        self.layoutWidget_6 = QWidget(self.targetGroup)
        self.layoutWidget_6.setObjectName(u"layoutWidget_6")
        self.layoutWidget_6.setGeometry(QRect(10, 220, 181, 121))
        self.verticalLayout_5 = QVBoxLayout(self.layoutWidget_6)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.label_42 = QLabel(self.layoutWidget_6)
        self.label_42.setObjectName(u"label_42")

        self.verticalLayout_5.addWidget(self.label_42)

        self.horizontalLayout_14 = QHBoxLayout()
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.label_43 = QLabel(self.layoutWidget_6)
        self.label_43.setObjectName(u"label_43")

        self.horizontalLayout_14.addWidget(self.label_43)

        self.TargetMinHSlider = QSlider(self.layoutWidget_6)
        self.TargetMinHSlider.setObjectName(u"TargetMinHSlider")
        self.TargetMinHSlider.setMaximum(255)
        self.TargetMinHSlider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_14.addWidget(self.TargetMinHSlider)

        self.label_44 = QLabel(self.layoutWidget_6)
        self.label_44.setObjectName(u"label_44")

        self.horizontalLayout_14.addWidget(self.label_44)


        self.verticalLayout_5.addLayout(self.horizontalLayout_14)

        self.horizontalLayout_15 = QHBoxLayout()
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.label_45 = QLabel(self.layoutWidget_6)
        self.label_45.setObjectName(u"label_45")

        self.horizontalLayout_15.addWidget(self.label_45)

        self.TargetMinSSlider = QSlider(self.layoutWidget_6)
        self.TargetMinSSlider.setObjectName(u"TargetMinSSlider")
        self.TargetMinSSlider.setMaximum(255)
        self.TargetMinSSlider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_15.addWidget(self.TargetMinSSlider)

        self.label_46 = QLabel(self.layoutWidget_6)
        self.label_46.setObjectName(u"label_46")

        self.horizontalLayout_15.addWidget(self.label_46)


        self.verticalLayout_5.addLayout(self.horizontalLayout_15)

        self.horizontalLayout_16 = QHBoxLayout()
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.label_47 = QLabel(self.layoutWidget_6)
        self.label_47.setObjectName(u"label_47")

        self.horizontalLayout_16.addWidget(self.label_47)

        self.TargetMinVSlider = QSlider(self.layoutWidget_6)
        self.TargetMinVSlider.setObjectName(u"TargetMinVSlider")
        self.TargetMinVSlider.setMaximum(255)
        self.TargetMinVSlider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_16.addWidget(self.TargetMinVSlider)

        self.label_48 = QLabel(self.layoutWidget_6)
        self.label_48.setObjectName(u"label_48")

        self.horizontalLayout_16.addWidget(self.label_48)


        self.verticalLayout_5.addLayout(self.horizontalLayout_16)

        self.layoutWidget_7 = QWidget(self.targetGroup)
        self.layoutWidget_7.setObjectName(u"layoutWidget_7")
        self.layoutWidget_7.setGeometry(QRect(11, 101, 181, 101))
        self.verticalLayout_6 = QVBoxLayout(self.layoutWidget_7)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.label_49 = QLabel(self.layoutWidget_7)
        self.label_49.setObjectName(u"label_49")

        self.verticalLayout_6.addWidget(self.label_49)

        self.horizontalLayout_17 = QHBoxLayout()
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.label_50 = QLabel(self.layoutWidget_7)
        self.label_50.setObjectName(u"label_50")

        self.horizontalLayout_17.addWidget(self.label_50)

        self.TargetMaxHSlider = QSlider(self.layoutWidget_7)
        self.TargetMaxHSlider.setObjectName(u"TargetMaxHSlider")
        self.TargetMaxHSlider.setMaximum(255)
        self.TargetMaxHSlider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_17.addWidget(self.TargetMaxHSlider)

        self.label_51 = QLabel(self.layoutWidget_7)
        self.label_51.setObjectName(u"label_51")

        self.horizontalLayout_17.addWidget(self.label_51)


        self.verticalLayout_6.addLayout(self.horizontalLayout_17)

        self.horizontalLayout_18 = QHBoxLayout()
        self.horizontalLayout_18.setObjectName(u"horizontalLayout_18")
        self.label_52 = QLabel(self.layoutWidget_7)
        self.label_52.setObjectName(u"label_52")

        self.horizontalLayout_18.addWidget(self.label_52)

        self.TargetMaxSSlider = QSlider(self.layoutWidget_7)
        self.TargetMaxSSlider.setObjectName(u"TargetMaxSSlider")
        self.TargetMaxSSlider.setMaximum(255)
        self.TargetMaxSSlider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_18.addWidget(self.TargetMaxSSlider)

        self.label_53 = QLabel(self.layoutWidget_7)
        self.label_53.setObjectName(u"label_53")

        self.horizontalLayout_18.addWidget(self.label_53)


        self.verticalLayout_6.addLayout(self.horizontalLayout_18)

        self.horizontalLayout_19 = QHBoxLayout()
        self.horizontalLayout_19.setObjectName(u"horizontalLayout_19")
        self.label_54 = QLabel(self.layoutWidget_7)
        self.label_54.setObjectName(u"label_54")

        self.horizontalLayout_19.addWidget(self.label_54)

        self.TargetMaxVSlider = QSlider(self.layoutWidget_7)
        self.TargetMaxVSlider.setObjectName(u"TargetMaxVSlider")
        self.TargetMaxVSlider.setMaximum(255)
        self.TargetMaxVSlider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_19.addWidget(self.TargetMaxVSlider)

        self.label_55 = QLabel(self.layoutWidget_7)
        self.label_55.setObjectName(u"label_55")

        self.horizontalLayout_19.addWidget(self.label_55)


        self.verticalLayout_6.addLayout(self.horizontalLayout_19)

        self.layoutWidget2 = QWidget(self.targetGroup)
        self.layoutWidget2.setObjectName(u"layoutWidget2")
        self.layoutWidget2.setGeometry(QRect(10, 30, 181, 61))
        self.verticalLayout_2 = QVBoxLayout(self.layoutWidget2)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.detectTargetCheckBox = QCheckBox(self.layoutWidget2)
        self.detectTargetCheckBox.setObjectName(u"detectTargetCheckBox")

        self.verticalLayout_2.addWidget(self.detectTargetCheckBox)

        self.getTargetCheckBox = QCheckBox(self.layoutWidget2)
        self.getTargetCheckBox.setObjectName(u"getTargetCheckBox")

        self.verticalLayout_2.addWidget(self.getTargetCheckBox)

        self.detectShotsCheckBox = QCheckBox(self.targetGroup)
        self.detectShotsCheckBox.setObjectName(u"detectShotsCheckBox")
        self.detectShotsCheckBox.setGeometry(QRect(10, 350, 181, 23))

        self.gridLayout_3.addWidget(self.scrollAreaFrame, 0, 0, 1, 1)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents_2)
        self.logGroup = QGroupBox(self.sideFrame)
        self.logGroup.setObjectName(u"logGroup")
        self.logGroup.setGeometry(QRect(0, 470, 231, 161))
        self.gridLayout_5 = QGridLayout(self.logGroup)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.textEdit = QTextEdit(self.logGroup)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setReadOnly(True)

        self.gridLayout_5.addWidget(self.textEdit, 0, 0, 1, 1)

        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setGeometry(QRect(230, 0, 1131, 631))
        self.camTab = QWidget()
        self.camTab.setObjectName(u"camTab")
        self.camSource = QLabel(self.camTab)
        self.camSource.setObjectName(u"camSource")
        self.camSource.setGeometry(QRect(0, 0, 800, 600))
        font = QFont()
        font.setPointSize(30)
        font.setBold(False)
        self.camSource.setFont(font)
        self.camSource.setAutoFillBackground(True)
        self.camSource.setAlignment(Qt.AlignCenter)
        self.groupBox = QGroupBox(self.camTab)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(810, 10, 321, 591))
        self.layoutWidget3 = QWidget(self.groupBox)
        self.layoutWidget3.setObjectName(u"layoutWidget3")
        self.layoutWidget3.setGeometry(QRect(10, 40, 301, 61))
        self.verticalLayout_3 = QVBoxLayout(self.layoutWidget3)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(self.layoutWidget3)
        self.label.setObjectName(u"label")

        self.verticalLayout_3.addWidget(self.label)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.minAreaSlider = QSlider(self.layoutWidget3)
        self.minAreaSlider.setObjectName(u"minAreaSlider")
        self.minAreaSlider.setMaximum(100000)
        self.minAreaSlider.setSingleStep(100)
        self.minAreaSlider.setOrientation(Qt.Horizontal)

        self.horizontalLayout.addWidget(self.minAreaSlider)

        self.label_2 = QLabel(self.layoutWidget3)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout.addWidget(self.label_2)


        self.verticalLayout_3.addLayout(self.horizontalLayout)

        self.tabWidget.addTab(self.camTab, "")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.maskSource = QLabel(self.tab)
        self.maskSource.setObjectName(u"maskSource")
        self.maskSource.setGeometry(QRect(0, 0, 800, 600))
        font1 = QFont()
        font1.setPointSize(30)
        self.maskSource.setFont(font1)
        self.maskSource.setAutoFillBackground(True)
        self.maskSource.setAlignment(Qt.AlignCenter)
        self.groupBox_2 = QGroupBox(self.tab)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(810, 10, 321, 591))
        self.layoutWidget4 = QWidget(self.groupBox_2)
        self.layoutWidget4.setObjectName(u"layoutWidget4")
        self.layoutWidget4.setGeometry(QRect(10, 30, 301, 91))
        self.verticalLayout_7 = QVBoxLayout(self.layoutWidget4)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.invertMaskCheckBox = QCheckBox(self.layoutWidget4)
        self.invertMaskCheckBox.setObjectName(u"invertMaskCheckBox")

        self.verticalLayout_7.addWidget(self.invertMaskCheckBox)

        self.erodeCheckBox = QCheckBox(self.layoutWidget4)
        self.erodeCheckBox.setObjectName(u"erodeCheckBox")

        self.verticalLayout_7.addWidget(self.erodeCheckBox)

        self.dialteCheckBox = QCheckBox(self.layoutWidget4)
        self.dialteCheckBox.setObjectName(u"dialteCheckBox")

        self.verticalLayout_7.addWidget(self.dialteCheckBox)

        self.tabWidget.addTab(self.tab, "")
        self.targetTab = QWidget()
        self.targetTab.setObjectName(u"targetTab")
        self.gridLayout_2 = QGridLayout(self.targetTab)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.targetFrame = QFrame(self.targetTab)
        self.targetFrame.setObjectName(u"targetFrame")
        self.targetFrame.setFrameShape(QFrame.StyledPanel)
        self.targetFrame.setFrameShadow(QFrame.Raised)
        self.targetSource = QLabel(self.targetFrame)
        self.targetSource.setObjectName(u"targetSource")
        self.targetSource.setGeometry(QRect(30, 60, 512, 512))
        self.targetSource.setFont(font1)
        self.targetSource.setAutoFillBackground(True)
        self.targetSource.setStyleSheet(u"")
        self.targetSource.setAlignment(Qt.AlignCenter)
        self.targetLabel = QLabel(self.targetFrame)
        self.targetLabel.setObjectName(u"targetLabel")
        self.targetLabel.setGeometry(QRect(30, 40, 67, 17))

        self.gridLayout_2.addWidget(self.targetFrame, 0, 0, 1, 1)

        self.targetMaskFrame = QFrame(self.targetTab)
        self.targetMaskFrame.setObjectName(u"targetMaskFrame")
        self.targetMaskFrame.setFrameShape(QFrame.StyledPanel)
        self.targetMaskFrame.setFrameShadow(QFrame.Raised)
        self.targetMaskSource = QLabel(self.targetMaskFrame)
        self.targetMaskSource.setObjectName(u"targetMaskSource")
        self.targetMaskSource.setGeometry(QRect(30, 60, 512, 512))
        self.targetMaskSource.setFont(font1)
        self.targetMaskSource.setAutoFillBackground(True)
        self.targetMaskSource.setStyleSheet(u"")
        self.targetMaskSource.setAlignment(Qt.AlignCenter)
        self.targetMaskLabel = QLabel(self.targetMaskFrame)
        self.targetMaskLabel.setObjectName(u"targetMaskLabel")
        self.targetMaskLabel.setGeometry(QRect(30, 40, 91, 17))

        self.gridLayout_2.addWidget(self.targetMaskFrame, 0, 1, 1, 1)

        self.tabWidget.addTab(self.targetTab, "")
        self.scoretab = QWidget()
        self.scoretab.setObjectName(u"scoretab")
        self.groupBox_3 = QGroupBox(self.scoretab)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setGeometry(QRect(10, 10, 531, 591))
        self.scoreSource = QLabel(self.groupBox_3)
        self.scoreSource.setObjectName(u"scoreSource")
        self.scoreSource.setGeometry(QRect(10, 30, 512, 512))
        font2 = QFont()
        font2.setPointSize(30)
        font2.setBold(False)
        font2.setItalic(False)
        font2.setStrikeOut(False)
        font2.setKerning(True)
        self.scoreSource.setFont(font2)
        self.scoreSource.setAutoFillBackground(True)
        self.scoreSource.setAlignment(Qt.AlignCenter)
        self.layoutWidget9 = QWidget(self.groupBox_3)
        self.layoutWidget9.setObjectName(u"layoutWidget9")
        self.layoutWidget9.setGeometry(QRect(20, 550, 491, 39))
        self.horizontalLayout_5 = QHBoxLayout(self.layoutWidget9)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.label_5 = QLabel(self.layoutWidget9)
        self.label_5.setObjectName(u"label_5")
        font3 = QFont()
        font3.setPointSize(20)
        font3.setBold(True)
        self.label_5.setFont(font3)
        self.label_5.setAutoFillBackground(True)
        self.label_5.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_5.addWidget(self.label_5)

        self.scoreValueLabel = QLabel(self.layoutWidget9)
        self.scoreValueLabel.setObjectName(u"scoreValueLabel")
        self.scoreValueLabel.setFont(font3)
        self.scoreValueLabel.setAutoFillBackground(True)
        self.scoreValueLabel.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_5.addWidget(self.scoreValueLabel)

        self.groupBox_4 = QGroupBox(self.scoretab)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.groupBox_4.setGeometry(QRect(550, 10, 281, 581))
        self.groupBox_5 = QGroupBox(self.scoretab)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.groupBox_5.setGeometry(QRect(840, 10, 281, 581))
        self.tabWidget_2 = QTabWidget(self.groupBox_5)
        self.tabWidget_2.setObjectName(u"tabWidget_2")
        self.tabWidget_2.setGeometry(QRect(0, 20, 281, 561))
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.layoutWidget11 = QWidget(self.tab_2)
        self.layoutWidget11.setObjectName(u"layoutWidget11")
        self.layoutWidget11.setGeometry(QRect(10, 10, 261, 121))
        self.verticalLayout_8 = QVBoxLayout(self.layoutWidget11)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.label_3 = QLabel(self.layoutWidget11)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_6.addWidget(self.label_3)

        self.serverHost = QLineEdit(self.layoutWidget11)
        self.serverHost.setObjectName(u"serverHost")

        self.horizontalLayout_6.addWidget(self.serverHost)


        self.verticalLayout_8.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.label_4 = QLabel(self.layoutWidget11)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_7.addWidget(self.label_4)

        self.serverPort = QLineEdit(self.layoutWidget11)
        self.serverPort.setObjectName(u"serverPort")

        self.horizontalLayout_7.addWidget(self.serverPort)


        self.verticalLayout_8.addLayout(self.horizontalLayout_7)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.startServerBtn = QPushButton(self.layoutWidget11)
        self.startServerBtn.setObjectName(u"startServerBtn")

        self.horizontalLayout_8.addWidget(self.startServerBtn)

        self.stopServerBtn = QPushButton(self.layoutWidget11)
        self.stopServerBtn.setObjectName(u"stopServerBtn")

        self.horizontalLayout_8.addWidget(self.stopServerBtn)


        self.verticalLayout_8.addLayout(self.horizontalLayout_8)

        self.calculateScoreFlag = QCheckBox(self.tab_2)
        self.calculateScoreFlag.setObjectName(u"calculateScoreFlag")
        self.calculateScoreFlag.setGeometry(QRect(10, 140, 0, 0))
        self.calculateScoreFlag.setIconSize(QSize(0, 0))
        self.calculateScoreFlag.setCheckable(True)
        self.calculateScoreFlag.setTristate(False)
        self.tabWidget_2.addTab(self.tab_2, "")
        self.tab_3 = QWidget()
        self.tab_3.setObjectName(u"tab_3")
        self.layoutWidget12 = QWidget(self.tab_3)
        self.layoutWidget12.setObjectName(u"layoutWidget12")
        self.layoutWidget12.setGeometry(QRect(10, 10, 261, 121))
        self.verticalLayout_9 = QVBoxLayout(self.layoutWidget12)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.verticalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.label_6 = QLabel(self.layoutWidget12)
        self.label_6.setObjectName(u"label_6")

        self.horizontalLayout_9.addWidget(self.label_6)

        self.clientHost = QLineEdit(self.layoutWidget12)
        self.clientHost.setObjectName(u"clientHost")

        self.horizontalLayout_9.addWidget(self.clientHost)


        self.verticalLayout_9.addLayout(self.horizontalLayout_9)

        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.label_7 = QLabel(self.layoutWidget12)
        self.label_7.setObjectName(u"label_7")

        self.horizontalLayout_10.addWidget(self.label_7)

        self.clientPort = QLineEdit(self.layoutWidget12)
        self.clientPort.setObjectName(u"clientPort")

        self.horizontalLayout_10.addWidget(self.clientPort)


        self.verticalLayout_9.addLayout(self.horizontalLayout_10)

        self.horizontalLayout_20 = QHBoxLayout()
        self.horizontalLayout_20.setObjectName(u"horizontalLayout_20")
        self.clientConnectBtn = QPushButton(self.layoutWidget12)
        self.clientConnectBtn.setObjectName(u"clientConnectBtn")

        self.horizontalLayout_20.addWidget(self.clientConnectBtn)

        self.clientDisconnectBtn = QPushButton(self.layoutWidget12)
        self.clientDisconnectBtn.setObjectName(u"clientDisconnectBtn")

        self.horizontalLayout_20.addWidget(self.clientDisconnectBtn)


        self.verticalLayout_9.addLayout(self.horizontalLayout_20)

        self.clientGetScoreBtn = QPushButton(self.tab_3)
        self.clientGetScoreBtn.setObjectName(u"clientGetScoreBtn")
        self.clientGetScoreBtn.setGeometry(QRect(10, 150, 89, 25))
        self.tabWidget_2.addTab(self.tab_3, "")
        self.tabWidget.addTab(self.scoretab, "")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.TargetMinHSlider.valueChanged.connect(self.label_44.setNum)
        self.TargetMinSSlider.valueChanged.connect(self.label_46.setNum)
        self.TargetMinVSlider.valueChanged.connect(self.label_48.setNum)
        self.TargetMaxHSlider.valueChanged.connect(self.label_51.setNum)
        self.TargetMaxSSlider.valueChanged.connect(self.label_53.setNum)
        self.TargetMaxVSlider.valueChanged.connect(self.label_55.setNum)
        self.minAreaSlider.valueChanged.connect(self.label_2.setNum)

        self.colorComboBox.setCurrentIndex(0)
        self.tabWidget.setCurrentIndex(3)
        self.tabWidget_2.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actionSettings.setText(QCoreApplication.translate("MainWindow", u"Settings", None))
        self.settingGroup.setTitle(QCoreApplication.translate("MainWindow", u"Settings", None))
        self.camGroup.setTitle(QCoreApplication.translate("MainWindow", u"Cam", None))
        self.camDeviceComboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"select camera device", None))

        self.startButton.setText(QCoreApplication.translate("MainWindow", u"start", None))
        self.stopButton.setText(QCoreApplication.translate("MainWindow", u"stop ", None))
        self.deviceLabel.setText(QCoreApplication.translate("MainWindow", u"Camera Device", None))
        self.maskGroup.setTitle(QCoreApplication.translate("MainWindow", u"Mask", None))
        self.getMaskCheckBox.setText(QCoreApplication.translate("MainWindow", u"Get Mask", None))
        self.colorComboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"RED", None))

        self.colorComboBox.setCurrentText(QCoreApplication.translate("MainWindow", u"RED", None))
        self.colorButton.setText(QCoreApplication.translate("MainWindow", u"choose color", None))
        self.targetGroup.setTitle(QCoreApplication.translate("MainWindow", u"Target", None))
        self.label_42.setText(QCoreApplication.translate("MainWindow", u"Min HSV", None))
        self.label_43.setText(QCoreApplication.translate("MainWindow", u"H", None))
        self.label_44.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_45.setText(QCoreApplication.translate("MainWindow", u"S", None))
        self.label_46.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_47.setText(QCoreApplication.translate("MainWindow", u"v", None))
        self.label_48.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_49.setText(QCoreApplication.translate("MainWindow", u"Max HSV", None))
        self.label_50.setText(QCoreApplication.translate("MainWindow", u"H", None))
        self.label_51.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_52.setText(QCoreApplication.translate("MainWindow", u"S", None))
        self.label_53.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_54.setText(QCoreApplication.translate("MainWindow", u"v", None))
        self.label_55.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.detectTargetCheckBox.setText(QCoreApplication.translate("MainWindow", u"Detect Target", None))
        self.getTargetCheckBox.setText(QCoreApplication.translate("MainWindow", u"Get Target", None))
        self.detectShotsCheckBox.setText(QCoreApplication.translate("MainWindow", u"Detect Shots", None))
        self.logGroup.setTitle(QCoreApplication.translate("MainWindow", u"Logs", None))
        self.textEdit.setPlaceholderText(QCoreApplication.translate("MainWindow", u"application logs", None))
        self.camSource.setText(QCoreApplication.translate("MainWindow", u"cam source", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Advance Settings", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Min Area", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.camTab), QCoreApplication.translate("MainWindow", u"Cam", None))
        self.maskSource.setText(QCoreApplication.translate("MainWindow", u"mask source", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"Advance Settings", None))
        self.invertMaskCheckBox.setText(QCoreApplication.translate("MainWindow", u"Invert Mask", None))
        self.erodeCheckBox.setText(QCoreApplication.translate("MainWindow", u"Erode", None))
        self.dialteCheckBox.setText(QCoreApplication.translate("MainWindow", u"Dialate", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("MainWindow", u"Mask", None))
        self.targetSource.setText(QCoreApplication.translate("MainWindow", u"target source", None))
        self.targetLabel.setText(QCoreApplication.translate("MainWindow", u"Target", None))
        self.targetMaskSource.setText(QCoreApplication.translate("MainWindow", u"target mask source", None))
        self.targetMaskLabel.setText(QCoreApplication.translate("MainWindow", u"Target Mask", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.targetTab), QCoreApplication.translate("MainWindow", u"Target", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"Shot Score", None))
        self.scoreSource.setText(QCoreApplication.translate("MainWindow", u"score source", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Score", None))
        self.scoreValueLabel.setText(QCoreApplication.translate("MainWindow", u"00.00", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("MainWindow", u"Metadata", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("MainWindow", u"Network", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Host :", None))
        self.serverHost.setPlaceholderText(QCoreApplication.translate("MainWindow", u"127.0.0.1", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Port :", None))
        self.serverPort.setPlaceholderText(QCoreApplication.translate("MainWindow", u"0000", None))
        self.startServerBtn.setText(QCoreApplication.translate("MainWindow", u"Start Server", None))
        self.stopServerBtn.setText(QCoreApplication.translate("MainWindow", u"Stop Server", None))
        self.calculateScoreFlag.setText("")
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_2), QCoreApplication.translate("MainWindow", u"Server", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Host :", None))
        self.clientHost.setPlaceholderText(QCoreApplication.translate("MainWindow", u"127.0.0.1", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Port :", None))
        self.clientPort.setPlaceholderText(QCoreApplication.translate("MainWindow", u"0000", None))
        self.clientConnectBtn.setText(QCoreApplication.translate("MainWindow", u"Connect", None))
        self.clientDisconnectBtn.setText(QCoreApplication.translate("MainWindow", u"Disconnect", None))
        self.clientGetScoreBtn.setText(QCoreApplication.translate("MainWindow", u"Get Score", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_3), QCoreApplication.translate("MainWindow", u"Client", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.scoretab), QCoreApplication.translate("MainWindow", u"Score", None))
    # retranslateUi

