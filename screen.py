from mss import mss
import cv2
import threading
import numpy as np
import mouse
import time
import pyautogui

class ScreenViewer:

    def __init__(self, resolution = (1920, 1080)):
        self.x_res = resolution[0]
        self.y_res = resolution[1]
        self.screen_resolution = {'top': 0, 'left': 0, 'width': self.x_res, 'height': self.y_res}

        # Snap Box Dimensions
        self.snap_square_length = 20

        # BGR Separators for Red
        self.red_min = 190
        self.red_max = 255
        
        self.green_max = 100
        self.blue_max = 100

        # Time
        self.last_shot_time = time.time()
        self.time_between_shots = 1.3
        self.can_shoot = True

        # Capturing
        self.capturing = True
        threading.Thread(target=self.start_capture_loop).start()

    def apply_transforms(self, image):
        image = image[:, :, :3] # Drop Alpha Channel
        # image = cv2.GaussianBlur(image, (5, 5), 0)
        # Filtered Red Locations
        image = cv2.inRange(image, (0, 0, self.red_min), (self.blue_max, self.green_max, self.red_max))
        
        return image

    def is_on_target(self, image):
        # See if the surrounding pixels are illuminated
        # cv2.imshow('Image', image)
        # cv2.waitKey(1)

        # Click
        image = image[round(self.y_res*99/200):round(self.y_res*101/200), round(self.x_res*99/200):round(self.x_res*101/200)]
        # Track
        # image = image[round(self.y_res*47/100):round(self.y_res*53/100), round(self.x_res*47/100):round(self.x_res*53/100)]

        image = image.flatten()
        # If a single pixel is illuminated, we are on the target
        for pixel in image:
            if pixel > 0:
                return True
        return False

    # def snap_to_target(self, image):
    #     center_x = round(self.x_res/2)
    #     center_y = round(self.y_res/2)
    #     top_left = (center_x - self.snap_square_length, center_y - self.snap_square_length)
    #     bottom_right = (center_x + self.snap_square_length, center_y + self.snap_square_length)
        
    #     image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    #     for i in range(len(image)):
    #         for pixel in image[i]:
    #             if pixel > 0:
    #                 pyautogui.move(center_x - top_left[0] + i, center_y - top_left[1] + i)
    #                 # self.shoot()
    #                 return
        

    def shoot(self):
        
        mouse.click(button='left')
        self.last_shot_time = time.time()
        self.can_shoot = False
    
    def press(self):
        pyautogui.mouseDown()
    
    def unpress(self):
        pyautogui.mouseUp()

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

                # Burst
                if is_shoot:
                    self.shoot()

                # # Click
                # if self.can_shoot:
                #     # Get Screen Frame
                #     frame = self.get_screen_frame()
                #     # Apply Transforms
                #     frame = self.apply_transforms(frame)
                #     # Predict and Click Mouse
                #     is_shoot = self.is_on_target(frame)

                #     # Burst
                #     if is_shoot:
                #         self.shoot()
                # elif self.last_shot_time + self.time_between_shots < time.time():
                #     # Set Boolean if Can Shoot Again
                #     self.can_shoot = True

                # # Track
                # # Get Screen Frame
                # frame = self.get_screen_frame()
                # # Apply Transforms
                # frame = self.apply_transforms(frame)
                # # Predict and Click Mouse
                # is_shoot = self.is_on_target(frame)

                # # Shoot If Is Shoot
                # if is_shoot:
                #     self.press()
                # else:
                #     self.unpress()

            cv2.waitKey(20)