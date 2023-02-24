import sys

import cv2
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

import mediapipe as mp


# Subclass QMainWindow to customize your application's main window 
class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow,self).__init__()
        
        self.setWindowTitle('DBT Fine Motor Analysis Demo')



        #set the layout for the window
        self.VBL = QVBoxLayout()
        #Populate the layout with widgets
        #Will be used to display image from webacm
        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)


        #Create instance of QThread class
        self.Worker1 = Worker1()
        #start the function and emit a signal
        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)

        #Cancel Button to stop videofeed
        self.CancelBTN = QPushButton('Stop Stream')
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.CancelBTN)


        self.setLayout(self.VBL)


    def ImageUpdateSlot(self,Image):
        #capture the emitted image from worker1 and update the FeedLabel
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.Worker1.stop()

class Worker1(QThread):
    #handle retrieving image from webcam
    ImageUpdate = pyqtSignal(QImage)
    def __init__(self):
        super().__init__()
        self.ThreadActive = True

    #will be called when we call the .start argument for the QThread object
    def run(self):
        # capture from webcam
        self.ThreadActive = True
        self.capture = cv2.VideoCapture(0)
        
        #This is the where we will run  the video and I guess do the processing
        #similar to the while loop in the TestBed notebook
        while self.ThreadActive:
            ret, frame = self.capture.read()
            if ret: 
                #convert to rgb format
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FlippedImg = cv2.flip(img, 1)
                #Convert to PIC image that PyQT can deal with
                ConvertToQtFormat = QImage(FlippedImg.data,FlippedImg.shape[1],FlippedImg.shape[0], QImage.Format.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640,480, Qt.AspectRatioMode.KeepAspectRatio)
                #this will send a signal to the MainWindow of the app, 
                # transmitting the data under Pic
                self.ImageUpdate.emit(Pic)
        # Shutdown capture system
        self.capture.release()


    def stop(self):
        self.ThreadActive = False
        # self.quit()




# You need one (and only one) QApplication instance per application.
# Pass in sys.argv to allow command line arguments for your app.
# If you know you won't use command line arguments QApplication([]) works too.


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())