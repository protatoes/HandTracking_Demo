import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import time
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk


import json
import os
import tempfile  # pylint: disable=unused-import
from typing import NamedTuple

# resources dependency
# undeclared dependency
from mediapipe.python.solutions import drawing_styles
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands
import cv2
import mediapipe as mp


class handDetector():
    def __init__(self,mode=False, maxHands = 2, detectionCon=0.5,trackCon=0.5,model_complexity=0):
        #initialize the configuration for mp Hands
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.model_complexity = model_complexity
        self.mpHands = mp.solutions.hands
        #Construct the hand detection model
        self.hands = self.mpHands.Hands(static_image_mode = self.mode,
                                        max_num_hands = self.maxHands,
                                        min_detection_confidence = self.detectionCon,
                                        min_tracking_confidence = self.trackCon,
                                        model_complexity = self.model_complexity)
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def _get_output_path(self, name):
            return os.path.join(tempfile.gettempdir(), self.id().split('.')[-1] + name)

    def _landmarks_list_to_array(self, landmark_list, image_shape):
        rows, cols, _ = image_shape
        return np.asarray([(lmk.x * cols, lmk.y * rows, lmk.z * cols)
                        for lmk in landmark_list.landmark])

    def _world_landmarks_list_to_array(self, landmark_list):
        return np.asarray([(lmk.x, lmk.y, lmk.z)
                        for lmk in landmark_list.landmark])


    def findHands(self, img,draw = True):
        #simple method for finding and displaying a hand on an image
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks: 
            for handLMS in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(img,
                                                   handLMS,
                                                   self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handID=0, draw = True):
        #list will have all the landmark positions
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handID] #[self.results.multi_hand_landmarks[id] for id in handID]
            
            for id, lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy])
                # if id == 4:
                if draw:
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
                
            
        return(lmList)
    
    def findRealWorldPositions(self, img, handID=0):
        lmList = []
        if self.results.multi_hand_world_landmarks:
            myHand = self.results.multi_hand_world_landmarks[handID]
            for id,hand_world_landmarks in enumerate(myHand.landmark):
                cx,cy = int(hand_world_landmarks.x),int(hand_world_landmarks.y)
                lmList.append([id,cx,cy])
        return(lmList)



class WebcamApp():
    def __init__(self, window, window_title):
        self.window = window
        self.window_title = window_title
        self.window.title(self.window_title)

        # Set up the webcam capture
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Set up the plot
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim([0, 10])
        self.xdata = np.arange(0, 100)
        self.ydata = np.zeros(100)
        self.line, = self.ax.plot(self.xdata, self.ydata)
        plt.ion()

        # Create the tkinter widgets
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack(side=tk.LEFT)
        self.plot_canvas = FigureCanvasTkAgg(self.fig, window)
        self.plot_canvas.get_tk_widget().pack(side=tk.RIGHT)

        # Create the "Start Test" button
        self.test_button = tk.Button(window, text="Start Test", command=self.start_test)
        self.test_button.pack(side=tk.BOTTOM)

        # Initialize the test variables
        self.is_testing = False
        self.test_data = []
        self.test_index = 0
        self.num_taps = 0
        self.last_tap_time = 0

        # Start the main loop
        self.window.after(0, self.update)
        self.window.mainloop()

    def start_test(self):
        if not self.is_testing:
            self.is_testing = True
            self.num_taps = 0
            self.last_tap_time = 0
            self.test_button.config(text="Stop Test")
        else:
            self.is_testing = False
            self.test_button.config(text="Start Test")
            self.update_plot()


    def update(self):
    # Get the next frame from the webcam
        ret, frame = self.capture.read()

        if ret:
            # Convert the frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the frame in the tkinter window
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, image=imgtk, anchor=tk.NW)

            # Update the plot with new data
            if self.is_testing:
                # Find the positions of the index and middle fingers
                hands = hand_tracker.detect_hands(frame)
                for hand in hands:
                    for finger in hand:
                        if finger['type'] == 'INDEX':
                            index_tip = finger['tip']
                        elif finger['type'] == 'MIDDLE':
                            middle_tip = finger['tip']

                # Check if the fingers have swapped places
                if index_tip[1] < middle_tip[1]:
                    if self.last_tap_time == 0 or time.time() - self.last_tap_time > 0.5:
                        self.num_taps += 1
                        self.last_tap_time = time.time()

                # Add the finger positions to the test data
                self.test_data.append((index_tip, middle_tip))

            if len(self.test_data) > 0 and self.test_index < len(self.ydata):
                self.ydata[self.test_index] = self.num_taps
                self.line.set_ydata(self.ydata)
                self.plot_canvas.draw()

                self.test_index += 1

        self.window.after(10, self.update)

    def quit(self):
        # Release the webcam capture resources
        self.capture.release()

        # Close the tkinter window
        self.window.destroy()