import sys
import matplotlib
import os
matplotlib.use('QtAgg')

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QFileDialog

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from GUI.interface import Ui_MainWindow
from algs import methods


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.setWindowTitle('gSpectrum')

        self.path1 = None
        self.path2 = None

        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.

        
        self.mfcc.clicked.connect(self.show_mfcc)
        self.melspec.clicked.connect(self.show_melspec)
        self.centroid.clicked.connect(self.show_centroid)
        self.hpss.clicked.connect(self.show_hpss)
        self.rolloff.clicked.connect(self.show_rolloff)
        self.chrom.clicked.connect(self.show_chrom)
        self.contrast.clicked.connect(self.show_contrast)
        self.tonnetz.clicked.connect(self.show_tonnetz)
        self.fft.clicked.connect(self.show_fft)
        self.zcr.clicked.connect(self.show_zcr)
        self.bendwith.clicked.connect(self.show_bendwith)

        self.getfile1.clicked.connect(self.getFileName1)
        self.getfile2.clicked.connect(self.getFileName2)

    def remove_widgets(self):
        for i in reversed(range(self.verticalLayout_2.count())): 
            self.verticalLayout_2.itemAt(i).widget().setParent(None)

    def show_mfcc(self):
        self.methods_obj = methods.jrock(self.path1, self.path2)
        self.remove_widgets()

        fig = self.methods_obj.mfcc()

        self.canvas = FigureCanvasQTAgg(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.verticalLayout_2.addWidget(self.toolbar)
        self.verticalLayout_2.addWidget(self.canvas)

    def show_melspec(self):
        self.methods_obj = methods.jrock(self.path1, self.path2)
        self.remove_widgets()

        fig = self.methods_obj.melspec()[0]

        self.canvas = FigureCanvasQTAgg(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.verticalLayout_2.addWidget(self.toolbar)
        self.verticalLayout_2.addWidget(self.canvas)

    def show_centroid(self):
        self.methods_obj = methods.jrock(self.path1, self.path2)
        self.remove_widgets()

        fig = self.methods_obj.centroid()[0]

        self.canvas = FigureCanvasQTAgg(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.verticalLayout_2.addWidget(self.toolbar)
        self.verticalLayout_2.addWidget(self.canvas)

    def show_hpss(self):
        self.methods_obj = methods.jrock(self.path1, self.path2)
        self.remove_widgets()

        fig = self.methods_obj.hpss()

        self.canvas = FigureCanvasQTAgg(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.verticalLayout_2.addWidget(self.toolbar)
        self.verticalLayout_2.addWidget(self.canvas)

    def show_rolloff(self):
        self.methods_obj = methods.jrock(self.path1, self.path2)
        self.remove_widgets()

        fig = self.methods_obj.rolloff()

        self.canvas = FigureCanvasQTAgg(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.verticalLayout_2.addWidget(self.toolbar)
        self.verticalLayout_2.addWidget(self.canvas)

    def show_chrom(self):
        self.methods_obj = methods.jrock(self.path1, self.path2)
        self.remove_widgets()

        fig = self.methods_obj.chromagram_func()

        self.canvas = FigureCanvasQTAgg(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.verticalLayout_2.addWidget(self.toolbar)
        self.verticalLayout_2.addWidget(self.canvas)

    def show_contrast(self):
        self.methods_obj = methods.jrock(self.path1, self.path2)
        self.remove_widgets()

        fig = self.methods_obj.contrast()

        self.canvas = FigureCanvasQTAgg(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.verticalLayout_2.addWidget(self.toolbar)
        self.verticalLayout_2.addWidget(self.canvas)

    def show_tonnetz(self):
        self.methods_obj = methods.jrock(self.path1, self.path2)
        self.remove_widgets()

        fig = self.methods_obj.tonnetz()

        self.canvas = FigureCanvasQTAgg(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.verticalLayout_2.addWidget(self.toolbar)
        self.verticalLayout_2.addWidget(self.canvas)

    def show_fft(self):
        self.methods_obj = methods.jrock(self.path1, self.path2)
        self.remove_widgets()

        fig = self.methods_obj.fft_func()

        self.canvas = FigureCanvasQTAgg(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.verticalLayout_2.addWidget(self.toolbar)
        self.verticalLayout_2.addWidget(self.canvas)

    def show_zcr(self):
        self.methods_obj = methods.jrock(self.path1, self.path2)
        self.remove_widgets()

        fig = self.methods_obj.zcr_func()[0]

        self.canvas = FigureCanvasQTAgg(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.verticalLayout_2.addWidget(self.toolbar)
        self.verticalLayout_2.addWidget(self.canvas)

    def show_bendwith(self):
        self.methods_obj = methods.jrock(self.path1, self.path2)
        self.remove_widgets()

        fig = self.methods_obj.spect_bendwidth_func()[0]

        self.canvas = FigureCanvasQTAgg(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.verticalLayout_2.addWidget(self.toolbar)
        self.verticalLayout_2.addWidget(self.canvas)

    def getFileName1(self):
        file_filter = 'Audio File (*.wav)'
        responce = QFileDialog.getOpenFileName(
            parent=self,
            caption='Select a file',
            directory=os.getcwd(),
            filter=file_filter,
            initialFilter='Audio File (*.wav)'
        )

        self.path1 = str(responce[0])

    
    def getFileName2(self):
        file_filter = 'Audio File (*.wav)'
        responce = QFileDialog.getOpenFileName(
            parent=self,
            caption='Select a file',
            directory=os.getcwd(),
            filter=file_filter,
            initialFilter='Audio File (*.wav)'
        )

        self.path2 = str(responce[0])


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()
    app.exec()
