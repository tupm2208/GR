# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI/main.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class MainUi(object):

    def __init__(self):
        self.is_updating = False
        self.trackers = None
        self.counter_trackers = None

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1021, 527)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.main_image = QtWidgets.QLabel(self.centralwidget)
        self.main_image.setGeometry(QtCore.QRect(0, 0, 854, 480))
        self.main_image.setObjectName("gate_label_image")
        self.main_face = QtWidgets.QLabel(self.centralwidget)
        self.main_face.setGeometry(QtCore.QRect(870, 0, 112, 112))
        self.main_face.setObjectName("label_7")
        self.formLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.formLayoutWidget_2.setGeometry(QtCore.QRect(860, 130, 131, 61))
        self.formLayoutWidget_2.setObjectName("formLayoutWidget_2")
        self.formLayout_2 = QtWidgets.QFormLayout(self.formLayoutWidget_2)
        self.formLayout_2.setContentsMargins(0, 0, 0, 0)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_8 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_8.setObjectName("label_8")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_8)
        self.label_9 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_9.setObjectName("label_9")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_9)
        self.label_10 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_10.setObjectName("label_10")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.label_10)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.formLayoutWidget_2)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_2)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(860, 200, 141, 21))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(860, 310, 141, 21))
        self.pushButton_2.setObjectName("pushButton_3")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(860, 420, 141, 21))
        self.pushButton_3.setObjectName("pushButton_3")
        self.formLayoutWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.formLayoutWidget_3.setGeometry(QtCore.QRect(860, 240, 141, 67))
        self.formLayoutWidget_3.setObjectName("formLayoutWidget_3")
        self.formLayout_3 = QtWidgets.QFormLayout(self.formLayoutWidget_3)
        self.formLayout_3.setContentsMargins(0, 0, 0, 0)
        self.formLayout_3.setObjectName("formLayout_3")
        self.label_11 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_11.setObjectName("label_11")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_11)
        self.label_12 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_12.setObjectName("label_12")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.label_12)
        self.label_13 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_13.setObjectName("label_13")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_13)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1021, 20))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.main_image.setText(_translate("MainWindow", "TextLabel"))
        self.main_face.setText(_translate("MainWindow", "gate_human_face"))
        self.label_8.setText(_translate("MainWindow", "ID:"))
        self.label_10.setText(_translate("MainWindow", "ID_value"))
        self.label_9.setText(_translate("MainWindow", "name:"))
        self.pushButton.setText(_translate("MainWindow", "Update Name"))
        self.pushButton_2.setText(_translate("MainWindow", "Update ID"))
        self.pushButton_3.setText(_translate("MainWindow", "Adjust"))
        self.label_11.setText(_translate("MainWindow", "ID:"))
        self.label_12.setText(_translate("MainWindow", "ID_value"))
        self.label_13.setText(_translate("MainWindow", "Old ID:"))

        self.pushButton_3.clicked.connect(self.toggle_status)
        self.pushButton.clicked.connect(self.update_name)
        self.pushButton_2.clicked.connect(self.update_id)

    def update_main_image(self, image):
        if self.is_updating:
            return
        import cv2
        tem = 'tem.jpg'
        cv2.imwrite(tem, image)

        img = QtGui.QPixmap(tem).scaledToHeight(480)
        # print(img.width(), img.height(), image.shape)

        self.main_image.setPixmap(img)

    def update_main_face(self, image):
        if self.is_updating:
            return
        if len(image) != 0:
            image = image[0]
        else:
            return
        # t1 = time()
        qimage = QtGui.QImage(image, image.shape[0], image.shape[1], image.strides[0], QtGui.QImage.Format_BGR888)

        img = QtGui.QPixmap(qimage)
        img.scaledToWidth(112)

        self.main_face.setPixmap(img)

    def set_timer(self, timer):
        self.timer = timer

    def toggle_status(self):

        self.formLayoutWidget_2.setVisible(not self.is_updating)
        self.pushButton.setVisible(not self.is_updating)
        self.formLayoutWidget_3.setVisible(not self.is_updating)
        self.pushButton_2.setVisible(not self.is_updating)

        if self.counter_trackers.first_tracker is not None:
            self.cr_track = self.counter_trackers.first_tracker

            self.cr_id = self.cr_track.get_identity()
            self.cr_id = str(self.cr_id)
            if self.cr_id.isdigit():
                self.cr_id = int(self.cr_id)
        self.is_updating = not self.is_updating

    def set_trackers(self, trackers):
        self.trackers = trackers

    def set_counter_trackers(self, trackers):
        self.counter_trackers = trackers

    def update_id(self):
        old_id = self.lineEdit_3.text()

        if not old_id.isdigit():
            print(old_id, 'invalid id')
            return
        self.trackers.update_id(self.cr_id, int(old_id))
        self.cr_track.original_names = [int(old_id)] * len(self.cr_track.original_names)
        self.toggle_status()

    def update_name(self):
        name = self.lineEdit_2.text()

        self.trackers.update_name(self.cr_id, name)

        self.toggle_status()

