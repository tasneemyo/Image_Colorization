# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'gui.ui'
##
## Created by: Qt User Interface Compiler version 6.10.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QMenuBar,
    QPushButton, QSizePolicy, QStatusBar, QWidget)

class Ui_MainWindow(object):
    
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1078, 667)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gray = QLabel(self.centralwidget)
        self.gray.setObjectName(u"gray")
        self.gray.setGeometry(QRect(40, 65, 431, 301))
        self.gray.setScaledContents(True)
        self.colorized_img = QLabel(self.centralwidget)
        self.colorized_img.setObjectName(u"colorized_img")
        self.colorized_img.setGeometry(QRect(580, 65, 431, 301))
        self.colorized_img.setScaledContents(True)
        self.upload = QPushButton(self.centralwidget)
        self.upload.setObjectName(u"upload")
        self.upload.setGeometry(QRect(110, 430, 100, 50))
        self.colorize = QPushButton(self.centralwidget)
        self.colorize.setObjectName(u"colorize")
        self.colorize.setGeometry(QRect(570, 430, 100, 50))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1078, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Image Colorization Project", None))
        self.gray.setText("")
        self.colorized_img.setText("")
        self.upload.setText(QCoreApplication.translate("MainWindow", u"Upload", None))
        self.colorize.setText(QCoreApplication.translate("MainWindow", u"Colorize", None))
    # retranslateUi

