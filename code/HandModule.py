import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import time
import os

import tempfile  # pylint: disable=unused-import

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