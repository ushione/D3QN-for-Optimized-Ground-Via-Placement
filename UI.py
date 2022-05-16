import sys
import torch
import argparse
import numpy as np
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from EvaluateNetwork import Build_Evaluate_Network


class Ui_MainWindow(QMainWindow):

    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.modelArray = np.zeros((10, 10))
        self.statusbar = QtWidgets.QStatusBar()
        self.menubar = QtWidgets.QMenuBar()
        self.centralwidget = QtWidgets.QWidget()
        self.changeFlag = 0
        self.setupUi(self)

    def setupUi(self, MainWindow):

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 500)
        self.setWindowIcon(QIcon('./Source/fish.ico'))
        self.centralwidget.setObjectName("centralwidget")
        self.icon1 = QIcon('./Source/1.png')
        self.icon2 = QIcon('./Source/2.png')
        self.icon3 = QIcon('./Source/3.png')
        self.row = 0
        self.col = 0
        self.buttonName = []
        for row in range(10):
            for col in range(10):
                num = row * 10 + col
                self.buttonName.append(num)
                self.buttonName[num] = QtWidgets.QPushButton(self.centralwidget)
                #
                self.buttonName[num].setGeometry(QtCore.QRect(40 * col + 70, 40 * row + 50, 40, 40))
                sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
                sizePolicy.setHorizontalStretch(1)
                sizePolicy.setVerticalStretch(1)
                sizePolicy.setHeightForWidth(self.buttonName[row].sizePolicy().hasHeightForWidth())
                self.buttonName[num].setSizePolicy(sizePolicy)
                self.buttonName[num].setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
                self.buttonName[num].setText('')
                self.buttonName[num].setIcon(self.icon1)
                self.buttonName[num].setIconSize(QtCore.QSize(38, 38))
                self.buttonName[num].setObjectName(str(row) + str(col))
                self.buttonName[num].setStyleSheet('border:2px solid rgb(0, 0, 0)')
                self.buttonName[num].clicked.connect(lambda: self.iconchange(self.sender().objectName()))
                #

        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(520, 200, 200, 65))
        self.textEdit.setObjectName("textEdit")
        self.textEdit.setReadOnly(True)

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(500, 300, 120, 40))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText('Calculate')
        self.pushButton.clicked.connect(self.calculate)

        self.clearButton = QtWidgets.QPushButton(self.centralwidget)
        self.clearButton.setGeometry(QtCore.QRect(660, 300, 80, 40))
        self.clearButton.setObjectName("clearButton")
        self.clearButton.setText('Clear')
        self.clearButton.clicked.connect(self.clear)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar.setObjectName("statusbar")

        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def draw_line(self):
        for i in range(100):
            self.buttonName[i].setStyleSheet('border:2px solid rgb(0, 0, 0)')

    def calculate(self):
        num1 = format(self.inputNet()[0][0], '.4f')
        self.textEdit.setText(str(num1))
        self.textEdit.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.textEdit.setFont(QtGui.QFont("微软雅黑", 20))

    def clear(self):
        self.modelArray = np.zeros((10, 10))
        for i_c in range(100):
            self.buttonName[i_c].setIcon(self.icon1)
        self.draw_line()
        self.textEdit.clear()
        print('Clear')

    def inputNet(self):
        x = torch.Tensor(self.modelArray.reshape(1, 100))
        print(netP(x).data.numpy())
        return netP(x).data.numpy()
        # return int(np.sum(self.modelArray))

    def iconchange(self, num):
        j = int(num) % 10
        i = int(num) // 10
        self.modelArray[i][j] = (self.modelArray[i][j] + 1) % 2
        if self.modelArray[i][j] == 1:
            self.buttonName[int(num)].setIcon(self.icon2)
        elif self.modelArray[i][j] == 0:
            self.buttonName[int(num)].setIcon(self.icon1)
        self.buttonName[int(num)].setIconSize(QtCore.QSize(38, 38))
        # self.draw_line()
        print(i, j, num)
        print(self.modelArray)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle("Fish")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UI of placing vias yourself.')
    parser.add_argument('-m', '--model_type', type=str, default='CNN_Inception', help='Please check EvaluateNetwork.py to choose your model.')
    args = parser.parse_args()
    model_type = args.model_type

    app = QApplication(sys.argv)
    w = Ui_MainWindow()
    w.show()

    netP = Build_Evaluate_Network(model_type)

    sys.exit(app.exec_())
