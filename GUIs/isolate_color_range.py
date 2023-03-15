#%%
import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2 as cv

# PyQt5 imports
from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import Qt as qt
from PyQt5 import uic


# Figure Imports
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True}) # to automatically fit long axis ticklabels


mw_Ui, mw_Base = uic.loadUiType('master_window.ui')
class MainGUI(mw_Base, mw_Ui):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.show()

        # Attributes
        self.img = None

        # Connect signals and slots
        self.openFileButton.clicked.connect(self.selectImage)
        self.hueMinSlider.valueChanged.connect(self.updateImage)
        self.hueMaxSlider.valueChanged.connect(self.updateImage)
        self.satMinSlider.valueChanged.connect(self.updateImage)
        self.satMaxSlider.valueChanged.connect(self.updateImage)
        self.valueMinSlider.valueChanged.connect(self.updateImage)
        self.valueMaxSlider.valueChanged.connect(self.updateImage)
        self.morphSlider.valueChanged.connect(self.updateImage)


    def selectImage(self, path=None):
        # self.path, _ = qtw.QFileDialog.getOpenFileName(self, "Choose File")
        self.path = 'mission ridge.jpg'
        self.filepath.setText(self.path)
        self.img = cv.imread(self.path)
        self.hsv = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
        return

    def updateImage(self):
        if self.img is None:
            return 
        
        # Read sliders
        hueMin, hueMax = self.hueMinSlider.value(), self.hueMaxSlider.value()
        satMin, satMax = self.satMinSlider.value(), self.satMaxSlider.value()
        valMin, valMax = self.valueMinSlider.value(), self.valueMaxSlider.value()
        pixels = self.morphSlider.value()

        # Error handling
        if hueMax<hueMin:
            hueMax = hueMin
            self.hueMaxSlider.setValue(hueMax)
        if satMax<satMin:
            satMax = satMin
            self.satMaxSlider.setValue(satMax)
        if valMax<valMin:
            valMax = valMin
            self.valueMaxSlider.setValue(valMax)

        # Display values
        self.hueMinOutput.setText(str(hueMin))
        self.hueMaxOutput.setText(str(hueMax))
        self.satMinOutput.setText(str(satMin))
        self.satMaxOutput.setText(str(satMax))
        self.valueMinOutput.setText(str(valMin))
        self.valueMaxOutput.setText(str(valMax))
        self.morphOutput.setText(str(pixels))

        # Compute mask and apply morph
        kernel = np.ones((pixels, pixels), np.uint8)
        mask = cv.inRange(self.hsv, (hueMin, satMin, valMin), (hueMax, satMax, valMax))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        masked_img = cv.bitwise_and(self.img, self.img, mask=mask)

        # Display image as label pixmap
        masked_img = cv.resize(masked_img, (800, 800))
        qImage = qtg.QImage(masked_img, masked_img.shape[1], masked_img.shape[0], qtg.QImage.Format_RGB888)
        pixmap = qtg.QPixmap.fromImage(qImage)
        self.imgLabel.setPixmap(pixmap)




if __name__ =='__main__':
    app = qtw.QApplication(sys.argv)
    gui = MainGUI()
    app.exec()
# %%
