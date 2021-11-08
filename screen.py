# Remove flickering mss
import mss.windows
mss.windows.CAPTUREBLT = True

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
    def __init__(self, name="OverShadow", uses_trigger_bot = False, uses_ammo_analyzer = True, resolution = (1920, 1080), time_between_frames_ms = 1):
        print("Initializing " + name + "...")
        self.name = name
        self.uses_trigger_bot = uses_trigger_bot
        self.uses_ammo_analyzer = uses_ammo_analyzer
        self.resolution = resolution
        self.time_between_frames_ms = time_between_frames_ms

        # Ammo Analyzer
        if uses_ammo_analyzer:
            self.ammo_analyzer = AmmoAnalyzer()
        else:
            self.ammo_analyzer = None
        
        # Trigger Bot
        if uses_trigger_bot:
            self.trigger_bot = TriggerBot()
        else:
            self.trigger_bot = None

        # MSS
        self.mss = mss()

        # UI Loop
        self.last_time = time.time()
        self.ui_loop = threading.Thread(target=self.ui_loop).start()

    def ui_loop(self):
        while True:
            # Get Screen
            sct_img = self.mss.grab({"top": 0, "left": 0, "width": self.resolution[0], "height": self.resolution[1]})
            img = np.array(sct_img)
            # img[:,:,2] = np.zeros([img.shape[0], img.shape[1]])

            # Draw UI
            # Ammo in middle of screen using opencv
            if self.uses_ammo_analyzer:
                if int(self.ammo_analyzer.current_ammo)<3: 
                    cv2.putText(img, str(self.ammo_analyzer.current_ammo), (self.resolution[0]//2+50, self.resolution[1]//2+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    cv2.putText(img, str(self.ammo_analyzer.current_ammo), (self.resolution[0]//2+50, self.resolution[1]//2+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                
            # Framerate Display
            cv2.putText(img, "FPS: " + str(round(1 / (time.time() - self.last_time), 2)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            self.last_time = time.time()

            # Show Screen
            cv2.imshow(self.name, img)
            # cv2.waitKey(self.time_between_frames_ms)
            cv2.waitKey(self.time_between_frames_ms)

        

class AmmoAnalyzer:
    def __init__(self, ammo_model_path='ammo_model.pkl', resolution = (1920, 1080), time_between_frames = 0.1):
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
        self.mss = mss()

        # Current Ammo
        self.current_ammo = '0'

        # Start Capture Loop
        threading.Thread(target=self.capture_loop).start()

    def get_frame(self):
        with self.mss as sct:
            # Get Screen Frame
            frame = np.array(sct.grab(self.monitor))
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
                # Wait
                time.sleep(self.time_between_frames)






class TrainAmmo:
    def __init__(self, data_path = 'ammo_frames/', model_path = 'ammo_model.pkl', train_test_split = 0.8, max_each_class = 1000):
        self.model = None
        self.model_path = model_path

        # Load labels and data locations
        self.train_data = []
        self.train_labels = []

        self.test_data = []
        self.test_labels = []

        # Read Data Locations
        self.train_data_read = []
        self.test_data_read = []

        # Data Path
        self.data_path = data_path

        # Max Each Class
        self.max_each_class = max_each_class

        # Train Test Split
        self.train_test_split = train_test_split
        
        self.classes = [0, 1, 2, 3, 4, 5, 6]
        self.label_dict = {'0': 0, '1': 1, '2': 2, '3' : 3, '4' : 4, '5' : 5, '6' : 6}

        # Load Data
        self.load_data()

        # Shuffle Data
        self.shuffle_data()

        # Test
        self.test_data = self.train_data[round(len(self.train_data)*self.train_test_split):]
        self.test_labels = self.train_labels[round(len(self.train_labels)*self.train_test_split):]
        # Train
        self.train_data = self.train_data[:round(len(self.train_data)*self.train_test_split)]
        self.train_labels = self.train_labels[:round(len(self.train_labels)*self.train_test_split)]

        # Read Data
        self.read_data()

        # Read Test Data
        self.read_test_data()

        # Train Model
        self.train()

    def apply_transformation(self, img):
        # Space for other transformations
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = img.reshape(-1)
        return img

    # Loads Labels and X Data Locations
    def load_data(self):
        # Glob folders in train folder
        folders = glob.glob(self.data_path + '/*')
        
        for folder in folders:
            # Get label
            label = folder.split('\\')[-1]
            print(label)
            label_id = self.label_dict[label]

            # Get image files
            image_files = glob.glob(folder + '/*.png')

            # Add data to x_data and y_labels
            added = 0
            for image_file in image_files:
                self.train_data.append(image_file)
                self.train_labels.append(label_id)
                added += 1
                if added>=self.max_each_class:
                    break

    # Shuffles data locations and labels together
    def shuffle_data(self):
        # Shuffle data
        # Zip x locations and y labels together and turn into a list
        combined = list(zip(self.train_data, self.train_labels))
        # Shuffle this list
        np.random.shuffle(combined)
        # Unzip the list after zipping it again
        self.train_data, self.train_labels = zip(*combined)
        
    def read_data(self):
        # Read Data
        counter = 0
        for data in self.train_data:
            img = cv2.imread(data)
            img = self.apply_transformation(img)
            self.train_data_read.append(img)
            counter += 1
            if counter % 100 == 0:
                print(counter)
    
    def read_test_data(self):
        counter = 0
        for data in self.test_data:
            img = cv2.imread(data)
            img = self.apply_transformation(img)
            self.test_data_read.append(img)
            counter += 1
            if counter % 100 == 0:
                print(counter)

    def train(self):
        # Model
        print('Training Model')
        # Shape of Data
        print(f'Data Shape: {np.array(self.train_data_read).shape}')
        print(f'Labels Shape: {np.array(self.train_labels).shape}')
        print(self.train_labels[:10])
        # # Logistic Regression
        self.model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1500, verbose=1)
        # Fit Data
        self.model.fit(self.train_data_read, self.train_labels)
        # Save model
        self.save_model()
        # Print Accuracy
        plot_confusion_matrix(self.model, self.test_data_read, self.test_labels)
        plt.show()
        self.test(self.model, self.test_data_read, self.test_labels)

    def test(self, model, test_x_final, test_y_final):
        preds = model.predict(test_x_final)
        correct = 0
        incorrect = 0
        for pred, gt in zip(preds, test_y_final):
            if pred == gt: correct += 1
            else: incorrect += 1
        print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")
        plt.show()
    
    def save_model(self):
        pickle.dump(self.model, open(self.model_path, 'wb'))

    def load_model(self):
        self.model = pickle.load(open(self.model_path, 'rb'))


class GatherData:
    def __init__(self, save_folder = f'ammo_frames/', resolution = (1920, 1080), time_between_frames = 0.05, start_file_number = 0):
        # Save Location
        self.save_folder = save_folder
        self.file_number = start_file_number

        # Resolution
        self.x_res = resolution[0]
        self.y_res = resolution[1]

        # Time Between Frames
        self.time_between_frames = time_between_frames

        # Initialize screen capture to ammo location
        top_left = (round(self.y_res*130/160), round(self.x_res*145/160))
        width = round(self.x_res*151/160) - top_left[1]
        height = round(self.y_res*138/160) - top_left[0]
        self.monitor = {'top': top_left[0], 'left': top_left[1], 'width': width, 'height': height}

        # Threshold Values
        self.threshold_low = 170
        self.threshold_high = 255

        # Initialize Capture Thread
        self.capture_thread = threading.Thread(target=self.capture_loop).start()

    def get_frame(self):
        with mss() as sct:
            # Get Screen Frame
            frame = np.array(sct.grab(self.monitor))
        return frame

    def process_frame_for_save(self, image):
        # Convert to Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Threshold
        # image = cv2.threshold(image, self.threshold_low, self.threshold_high, cv2.THRESH_BINARY)[1]
        # Resize
        image = cv2.resize(image, (28, 28))
        # Return
        return image

    def save_frame(self, frame):
        # Save Frame
        cv2.imwrite(f'{self.save_folder}ammo{self.file_number}.png', frame)
        self.file_number += 1

    # Main Data Capture Loop
    def capture_loop(self):
        while True:
            # Get Frame
            frame = self.get_frame()
            # Process Frame
            frame = self.process_frame_for_save(frame)
            # Save Frame
            self.save_frame(frame)
            # Wait
            time.sleep(self.time_between_frames)


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

        # MSS
        self.mss = mss()

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
                    self.shoot()