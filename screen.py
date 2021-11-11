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
import keyboard

class Window:
    def __init__(self, name="OverShadow", uses_trigger_bot = False, uses_ammo_analyzer = True, uses_flashbang_analyzer = True, uses_character_identifier = True, resolution = (1920, 1080), time_between_frames_ms = 1, ui_dimensions = (1920, 1080)):
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

        # # Character Identifier
        # if uses_character_identifier:
        #     self.character_identifier = CharacterIdentifier(mss=self.mss, resolution=self.resolution)
        # else:
        #     self.character_identifier = None

        # UI Dimensions
        self.ui_x = ui_dimensions[0]
        self.ui_y = ui_dimensions[1]

        # UI Loop
        self.last_time = time.time()
        self.ui_loop = threading.Thread(target=self.ui_loop).start()

    # def ui_loop(self):
    #     monitor = {"top": 0, "left": 0, "width": self.ui_x, "height": self.ui_y}
    #     while True:
    #         frame = np.array(self.mss.grab(monitor))

    #         # Sliding Window for Character Identifier
    #         drawn_image = self.character_identifier.sliding_window_draw(frame)
    #         cv2.imshow(self.name, drawn_image)
    #         cv2.waitKey(self.time_between_frames_ms)


    # Draws the UI
    def ui_loop(self):
        time.sleep(3)
        while True:
            # Green 500 by 500 box opencv
            image = np.zeros((self.ui_y, self.ui_x, 3), np.uint8)
            
            # Set Color of Background
            if self.flashbang_analyzer.available:
                cv2.rectangle(image, (0, 0), (self.ui_x, self.ui_y), (0, 255, 0), -1)
            else:
                cv2.rectangle(image, (0, 0), (self.ui_x, self.ui_y), (0, 0, 255), -1)

            # Put Ammo Text in Center
            if self.uses_ammo_analyzer:
                cv2.putText(image, "Ammo: " + str(self.ammo_analyzer.current_ammo), (self.ui_x//3, self.ui_y//2), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 3)

            cv2.imshow(self.name, image)
            cv2.waitKey(self.time_between_frames_ms)
        

class CharacterIdentifier:
    def __init__(self, model_path='character_identifier.pkl', resolution = (1920, 1080), time_between_frames = 0.1, mss=None, sliding_window_size = (1920//3, 1080//3)):
        self.model = self.load_model(model_path)
        # Resolution
        self.x_res = resolution[0]
        self.y_res = resolution[1]

        # Sliding Window
        self.sliding_window_size = sliding_window_size

        # Time between frames
        self.time_between_frames = time_between_frames

        # Initialize screen capture to ammo location
        self.top_left = (round(self.x_res/3), round(self.y_res/3))
        self.bottom_right = (round(self.x_res*2/3), round(self.y_res*2/3))
        self.monitor = {"top": self.top_left[1], "left": self.top_left[0], "width": self.bottom_right[0] - self.top_left[0], "height": self.bottom_right[1] - self.top_left[1]}

        
        # BGR Separators for Red
        self.red_min = 180
        self.red_max = 255
        self.green_max = 100
        self.blue_max = 100

        # Prediction String
        self.prediction_string = "None"

    # Apply Transformations before Network Input
    def apply_transformation(self, frame):
        # Drop Alpha Channel
        frame = frame[:, :, :3]
        # Filtered Red Locations
        frame = cv2.inRange(frame, (0, 0, self.red_min), (self.blue_max, self.green_max, self.red_max))
        return frame

    def predict(self, screenshot):
        processed_image = self.apply_transformation(screenshot)
        return self.model.predict(processed_image.reshape(1, -1))[0]

    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model

    def sliding_window_draw(self, image):
        original_image = image.copy()
        image = self.apply_transformation(image)
        # Crop
        image = image[self.top_left[1]:self.bottom_right[1], self.top_left[0]:self.bottom_right[0]]
        # Predict
        prediction = self.model.predict(image.reshape(1, -1))[0]
        if prediction == 0:
            self.prediction_string = "Tracer"
        elif prediction == 1:
            self.prediction_string = "Ashe"
        else:
            self.prediction_string = "None"

        if self.prediction_string != "None":
            cv2.putText(original_image, self.prediction_string, (self.top_left[0], self.top_left[1]), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        # # Sliding Window for Character Identifier
        # for x in range(0, self.x_res, self.sliding_window_size[0]):
        #     for y in range(0, self.y_res, self.sliding_window_size[1]):
        #         prediction = self.model.predict(image[y:y+self.sliding_window_size[1], x:x+self.sliding_window_size[0]].reshape(1, -1))[0]
        #         if prediction == 0:
        #             cv2.putText(original_image, "TRACER", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)
        #         elif prediction == 1:
        #             cv2.putText(original_image, "ASHE", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)
        #         cv2.rectangle(original_image, (x, y), (x+self.sliding_window_size[0], y+self.sliding_window_size[1]), (0, 255, 0), 5)
                
        
        return original_image





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

    def __init__(self, resolution = (1920, 1080), snap_square_length = 3, fire_delay=0.2, mss=None):
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

    def toggle_hud(self):
        # Press ALT+Z to toggle HUD
        keyboard.press_and_release('alt+z')
    
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