import time
import sys

import cv2
import numpy as np
import pandas as pd
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

        #Data Holder
        self.hand_dict_res =  pd.DataFrame(columns=['handIndex',"hand_label",'timestamp','finger', 'x', 'y','z'])
        self.record_data = False
        self.starttime = 0
        #instantiate mediapipe model
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode = False,
                                        max_num_hands = 2,
                                        min_detection_confidence = 0.4,
                                        min_tracking_confidence = 0.4, #tracking confidence will allow for the same hand to be tracked throughout
                                        model_complexity = 1)
        self.mpDraw = mp.solutions.drawing_utils

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
        self.starttime = time.time()

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
        cv_img = self.hand_tracking(cv_img)      
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio)
        #p = convert_to_Qt_format.scaled(801, 801, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
    def hand_tracking(self,img):
        #Instantiate handtracking model
        img.flags.writeable = False
        #convert image to RGB, mediapipe only takes rgb images
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #get detection,tracking results 
        results = self.hands.process(imgRGB)
        img.flags.writeable = True
        #check if a hand is detected
        if results.multi_hand_landmarks:
            #iterate over each hand and extract landmark information
            for handLms in results.multi_hand_landmarks:
                #get the hand_index and label
                handIndex = results.multi_hand_landmarks.index(handLms)
                hand_label = results.multi_handedness[handIndex].classification[0].label
                #get finger index and finger landmarks per hand
                time_c = time.time()
                for id,lm in enumerate(handLms.landmark):
                    #get image size to get the center of the landmarks in pixels
                    h,w,c = img.shape
                    cx,cy,cz = int(lm.x*w),int(lm.y*h),int(lm.z*w) 
                    #get a particular landmark 
                    if id == 8 or id == 12:
                        cv2.circle(img, (cx,cy),15,(255,0,0),cv2.FILLED)
                        row_dict = {"handIndex":handIndex,
                            "hand_label" : hand_label,
                            "timestamp": time_c - self.starttime,
                            "finger": id,
                            'x': cx,
                            'y':cy,
                            'z':cz}
                        if self.record_data:
                            self.hand_dict_res = self.hand_dict_res.append(row_dict, ignore_index=True)



        return img

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec())