
import sys

import cv2
import numpy as np
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

import mediapipe as mp
import matplotlib.pyplot as plt



class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # capture from web cam
        self._run_flag = True
        self.cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = self.cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        self.cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False


class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        # self.available_cameras = QCameraInfo.availableCameras()  # Getting available cameras

        cent = QGuiApplication.primaryScreen().availableGeometry().center()  # Finds the center of the screen
        self.setStyleSheet("background-color: white;")
        self.resize(1400, 800)
        self.frameGeometry()
        self.setWindowTitle('DBT Fine Motor Analysis Demo')
        self.initWindow()

########################################################################################################################
#                                                   Windows                                                            #
########################################################################################################################
    def initWindow(self):
        # create the video capture thread
        self.thread = VideoThread()

        # Button to start video
        self.ss_video = QPushButton(self)
        self.ss_video.setText('Start video')
        self.ss_video.move(769, 100)
        self.ss_video.resize(300, 100)
        self.ss_video.clicked.connect(self.ClickStartVideo)

        # Status bar
        self.status = QStatusBar()
        self.status.setStyleSheet("background : lightblue;")  # Setting style sheet to the status bar
        self.setStatusBar(self.status)  # Adding status bar to the main window
        self.status.showMessage('Ready to start')

        self.image_label = QLabel(self)
        self.disply_width = 669
        self.display_height = 501
        self.image_label.resize(self.disply_width, self.display_height)
        self.image_label.setStyleSheet("background : black;")
        self.image_label.move(0, 0)

########################################################################################################################
#                                                   Buttons                                                            #
########################################################################################################################
    # Activates when Start/Stop video button is clicked to Start (ss_video
    def ClickStartVideo(self):
        # Change label color to light blue
        self.ss_video.clicked.disconnect(self.ClickStartVideo)
        self.status.showMessage('Video Running...')
        # Change button to stop
        self.ss_video.setText('Stop video')
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)

        # start the thread
        self.thread.start()
        self.ss_video.clicked.connect(self.thread.stop)  # Stop the video if button clicked
        self.ss_video.clicked.connect(self.ClickStopVideo)

    # Activates when Start/Stop video button is clicked to Stop (ss_video)
    def ClickStopVideo(self):
        self.thread.change_pixmap_signal.disconnect()
        self.ss_video.setText('Start video')
        self.status.showMessage('Ready to start')
        self.ss_video.clicked.disconnect(self.ClickStopVideo)
        self.ss_video.clicked.disconnect(self.thread.stop)
        self.ss_video.clicked.connect(self.ClickStartVideo)

########################################################################################################################
#                                                   Actions                                                            #
########################################################################################################################

    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""

        cv_img = cv2.flip(cv_img, 1)        
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio)
        #p = convert_to_Qt_format.scaled(801, 801, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec())