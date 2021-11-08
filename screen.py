# Imports
from mss import mss
import cv2
import threading
import numpy as np
import mouse
import pyautogui
import time
import matplotlib as plt
import glob
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageGrab

class Window:
    def __init__(self, name="OverShadow", uses_trigger_bot = False, uses_ammo_analyzer = True, uses_flashbang_analyzer = True, resolution = (1920, 1080), time_between_frames_ms = 1):
        print("Initializing " + name + "...")
        self.name = name
        self.uses_trigger_bot = uses_trigger_bot
        self.uses_ammo_analyzer = uses_ammo_analyzer
        self.uses_flashbang_analyzer = uses_flashbang_analyzer
        self.resolution = resolution
        self.time_between_frames_ms = time_between_frames_ms

        # MSS
        self.mss = mss()

        # Ammo Analyzer
        if uses_ammo_analyzer:
            self.ammo_analyzer = AmmoAnalyzer(mss=self.mss, resolution=self.resolution)
        else:
            self.ammo_analyzer = None
        
        # Trigger Bot
        if uses_trigger_bot:
            self.trigger_bot = TriggerBot(mss=self.mss, resolution=self.resolution)
        else:
            self.trigger_bot = None

        # Flashbang Analyzer
        if uses_flashbang_analyzer:
            self.flashbang_analyzer = FlashbangAnalyzer(mss=self.mss, resolution=self.resolution)
        else:
            self.flashbang_analyzer = None

        # UI Loop
        self.last_time = time.time()
        self.ui_loop = threading.Thread(target=self.ui_loop).start()

    def ui_loop(self):
        time.sleep(3)
        while True:
            # Green 500 by 500 box opencv
            image = np.zeros((500, 500, 3), np.uint8)

            # Set Color of Background
            if self.flashbang_analyzer.available:
                cv2.rectangle(image, (0, 0), (500, 500), (0, 255, 0), -1)
            else:
                cv2.rectangle(image, (0, 0), (500, 500), (0, 0, 255), -1)

            # Put Ammo Text in Center
            if self.uses_ammo_analyzer:
                cv2.putText(image, "Ammo: " + str(self.ammo_analyzer.current_ammo), (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            cv2.imshow(self.name, image)
            cv2.waitKey(self.time_between_frames_ms)
        
class FlashbangAnalyzer:
    def __init__(self, ammo_model_path='flashbang_model.pkl', resolution = (1920, 1080), time_between_frames = 0.1, mss=None):
        self.model = self.load_model(ammo_model_path)
        # Resolution
        self.x_res = resolution[0]
        self.y_res = resolution[1]

        # Time between frames
        self.time_between_frames = time_between_frames

        # Initialize screen capture to ammo location
        top_left = (round(self.y_res*135/160), round(self.x_res*134/160))
        width = round(self.x_res*140/160) - top_left[1]
        height = round(self.y_res*145/160) - top_left[0]
        self.monitor = {'top': top_left[0], 'left': top_left[1], 'width': width, 'height': height}

        # MSS
        self.mss = mss

        # Current Ammo
        self.available = True

        # Start Capture Loop
        threading.Thread(target=self.capture_loop).start()

    def get_frame(self):
        # Get Screen Frame
        frame = np.array(self.mss.grab(self.monitor))
        return frame

    # Apply Transformations before Network Input
    def apply_transformation(self, image):
        # Convert to Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Resize
        image = cv2.resize(image, (28, 28))
        # Return
        return image

    def predict(self, screenshot):
        processed_image = self.apply_transformation(screenshot)
        return self.model.predict(processed_image.reshape(1, -1))[0]

    def load_model(self, ammo_model_path):
        with open(ammo_model_path, 'rb') as f:
            model = pickle.load(f)
        return model

    def capture_loop(self):
        with mss() as sct:
            while True:
                # Get Screenshot
                frame = self.get_frame()
                # Predict
                prediction = self.predict(frame)
                # Update Member Variable
                if prediction == 0:
                    self.available = True
                else:
                    self.available = False
                # Wait
                time.sleep(self.time_between_frames)

class AmmoAnalyzer:
    def __init__(self, ammo_model_path='ammo_model.pkl', resolution = (1920, 1080), time_between_frames = 0.1, mss=None):
        self.model = self.load_model(ammo_model_path)
        # Resolution
        self.x_res = resolution[0]
        self.y_res = resolution[1]

        # Time between frames
        self.time_between_frames = time_between_frames

        # Initialize screen capture to ammo location
        top_left = (round(self.y_res*130/160), round(self.x_res*145/160))
        width = round(self.x_res*151/160) - top_left[1]
        height = round(self.y_res*138/160) - top_left[0]
        self.monitor = {'top': top_left[0], 'left': top_left[1], 'width': width, 'height': height}

        # MSS
        self.mss = mss

        # Current Ammo
        self.current_ammo = '0'
        self.ammo_history = []
        self.ammo_history_buffer = 10

        # Start Capture Loop
        threading.Thread(target=self.capture_loop).start()

    def get_frame(self):
        # Get Screen Frame
        frame = np.array(self.mss.grab(self.monitor))
        return frame

    # Apply Transformations before Network Input
    def apply_transformation(self, image):
        # Convert to Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Resize
        image = cv2.resize(image, (28, 28))
        # Return
        return image

    def predict(self, screenshot):
        processed_image = self.apply_transformation(screenshot)
        return str(self.model.predict(processed_image.reshape(1, -1))[0])

    def load_model(self, ammo_model_path):
        with open(ammo_model_path, 'rb') as f:
            model = pickle.load(f)
        return model

    def capture_loop(self):
        with mss() as sct:
            while True:
                # Get Screenshot
                frame = self.get_frame()
                # Predict
                prediction = self.predict(frame)
                # Update Member Variable
                self.current_ammo = prediction
                # Ammo History
                self.ammo_history.append(int(prediction))
                if len(self.ammo_history) > self.ammo_history_buffer:
                    self.ammo_history.pop(0)
                # Wait
                time.sleep(self.time_between_frames)

class TriggerBot:

    def __init__(self, resolution = (1920, 1080), snap_square_length = 3, fire_delay=0.5, mss=None):
        self.x_res = resolution[0]
        self.y_res = resolution[1]

        # Snap Box Dimensions
        self.snap_square_length = snap_square_length
        # self.screen_resolution = {'top': 0, 'left': 0, 'width': self.x_res, 'height': self.y_res}
        self.screen_resolution = {'top': round(self.y_res/2)-self.snap_square_length, 'left': round(self.x_res/2)-self.snap_square_length, 'width': self.snap_square_length*2, 'height': self.snap_square_length*2}

        # BGR Separators for Red
        self.red_min = 180
        self.red_max = 255
        self.green_max = 100
        self.blue_max = 100

        # MSS
        self.mss = mss

        # Fire Delay
        self.fire_delay = fire_delay

        # Sees Person
        self.sees_person = False

        # Capturing
        self.capturing = True
        threading.Thread(target=self.start_capture_loop).start()

    def apply_transforms(self, image):
        image = image[:, :, :3] # Drop Alpha Channel
        # Filtered Red Locations
        image = cv2.inRange(image, (0, 0, self.red_min), (self.blue_max, self.green_max, self.red_max))
        return image

    def is_on_target(self, image):
        # See if the surrounding pixels are illuminated
        image = image.flatten()
        # Check if single pixel is illuminated
        if np.sum(image) > 0:
            return True
        return False

    def shoot(self):  
        mouse.click(button='left')
        # Time betwen shots -> (Should just sleep this thread -> only triggerbot)
        time.sleep(self.fire_delay)
    
    # def press(self):
    #     pyautogui.mouseDown()
    
    # def unpress(self):
    #     pyautogui.mouseUp()

    def get_screen_frame(self):
        with self.mss as sct:
            # Get Screen Frame
            frame = np.array(sct.grab(self.screen_resolution))
        return frame

    def start_capture(self):
        self.capturing = True

    def end_capture(self):
        self.capturing = False

    def start_capture_loop(self):
        # Get Current Screen Frame
        while True:
            if self.capturing:
                
                # Click No Wait
                # Get Screen Frame
                frame = self.get_screen_frame()
                # Apply Transforms
                frame = self.apply_transforms(frame)

                # Predict and Click Mouse
                is_shoot = self.is_on_target(frame)

                # # Burst
                if is_shoot:
                    self.sees_person = True
                    self.shoot()
                else:
                    self.sees_person = False