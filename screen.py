from mss import mss
import cv2
import threading
import numpy as np
import mouse
import pyautogui
import time

class TriggerBot:

    def __init__(self, resolution = (1920, 1080), snap_square_length = 3, fire_delay=0.5):
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

        # Fire Delay
        self.fire_delay = fire_delay

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
        with mss() as sct:
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
                    self.shoot()