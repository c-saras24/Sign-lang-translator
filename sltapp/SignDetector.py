# --------------------------------------------------
# importing the libraries and packages
# --------------------------------------------------
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import pyttsx3
try:
    from tkinter import *
except:
    from Tkinter import *
from tkinter import messagebox

import PIL.Image
import PIL.ImageTk
from tkinter import ttk
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array 
import tensorflow
import pyttsx3
engine = pyttsx3.init()
  
# This module is imported so that we can 
# play the converted audio
# --------------------------------------------------------------------------------------------------
# Class that contains all the class variables and methods which are used to create user interface
# --------------------------------------------------------------------------------------------------


class SignDetector:

    # --------------------------------------------------
    # Initializing class variables and UI components
    # --------------------------------------------------

    # main window
    __window = Tk()


    # Frames in the main window
    __canvas = Canvas(__window, width=450, height=400)
    __thisframe = Frame(__window, width=450, height=400,
                        highlightbackground='black', highlightthickness=1, bg='black')
    __canvas1 = Canvas(__thisframe, width=270, height=270, bg="black")

    # Images used for button widgets
    __photo1 = PhotoImage(file=r"C:/Users/user/OneDrive/Desktop/slt/sltapp/buttonimg/start.png")
    __photo3 = PhotoImage(file=r"C:/Users/user/OneDrive/Desktop/slt/sltapp/buttonimg/close.png")


    # ----------------------------------------------------------
    # Constructor method to initialize and place UI components
    # ----------------------------------------------------------

    def __init__(self):
        
        # Window properties
        self.__window.title('Signs To Heart')
        self.__window.configure(bg='#0047AB')
        self.__window.geometry("450x200+0+0")
        self.__window.resizable(False, False)

        # Built-in device camera
        self.vid = cv2.VideoCapture(0)
        # initialize weight for running average
        self.aWeight = 0.5
        # initialize num of frames
        self.num_frames = 0

        self.run_once = 0
        self.bg = None

        self.classifier = load_model('gesture.h5')


        # Buttons Used
        self.__b1 = Button(self.__window, text='Scan', image=self.__photo1,
                           bg='#0047AB', activebackground='#F0FFFF', bd=0, command=self.on_start)
        self.__b1.place(x=160, y=45)
        self.__b3 = Button(self.__window, text='Scan', image=self.__photo3,
                           bg='#0047AB', activebackground='#F0FFFF', bd=0, command=self.on_stop)
        self.__b3.place(x=159, y=150)

    # --------------------------------------------------
    # To find the running average over the background
    # --------------------------------------------------
    def run_avg(self, image, aWeight):
        """ To find the running average over the background. """

        # initialize the background
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return

        # compute weighted average, accumulate it and update the background
        cv2.accumulateWeighted(image, self.bg, aWeight)



    def on_closing(self):
        self.vid.release()
        cv2.destroyAllWindows()
        self.__window.destroy()
    
    # --------------------------------------------------
    # To start the application 
    # --------------------------------------------------

    def on_start(self):
        if self.run_once == 0:
            self.delay = 10
            self.update()
            self.run_once += 1
    
    # --------------------------------------------------
    # To ask user if he/she wants to quit application 
    # --------------------------------------------------

    def on_stop(self):
        if self.run_once > 0:
            if messagebox.askokcancel("Quit", "Do you want to quit?"):    
                self.vid.release()
                cv2.destroyAllWindows()
                self.__window.destroy()

   
    # -------------------------------------------------------
    # To update user interface continuously and predict gestures
    # -------------------------------------------------------

    def update(self):
        mp_holistic = mp.solutions.holistic # Holistic model
        mp_drawing = mp.solutions.drawing_utils # Drawing utilities
        def mediapipe_detection(image, model):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
            image.flags.writeable = False                  # Image is no longer writeable
            results = model.process(image)                 # Make prediction
            image.flags.writeable = True                   # Image is now writeable 
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
            return image, results
        def draw_landmarks(image, results):
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
        def draw_styled_landmarks(image, results):
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
        def extract_keypoints(results):
            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
            return np.concatenate([lh, rh])
        
        actions = np.array(["hello","thanks","i love you"]) 
        model = load_model('gesture.h5')
        sequence = []
        sentence = []
        predictions = []
        threshold = 0.7

        cap = cv2.VideoCapture(0)
# Set mediapipe model 
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                engine = pyttsx3.init()
                engine.setProperty("rate", 300)
                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                print(results)
                
                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # 2. Prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]
                
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    predictions.append(np.argmax(res))
                    
                    
                #3. Viz logic
                    if np.unique(predictions[-10:])[0]==np.argmax(res): 
                        if res[np.argmax(res)] > threshold: 
                            
                            if len(sentence) > 0: 
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                                    engine.say(actions[np.argmax(res)])
                                    engine.runAndWait()
                            else:
                                sentence.append(actions[np.argmax(res)])
                                engine.say(actions[np.argmax(res)])
                                engine.runAndWait()

                    if len(sentence) > 5: 
                        sentence = sentence[-5:]

                    
                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Show to screen
                cv2.imshow('OpenCV Feed', image)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()    
       
       
        
    # -------------------------------------------------------
    # To run the application
    # -------------------------------------------------------

    def run(self):
        self.__window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.__window.mainloop()
        

detector = SignDetector()
detector.run()
