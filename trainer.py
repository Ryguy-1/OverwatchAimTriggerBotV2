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

class Train:
    def __init__(self, data_path = 'flashbang_frames/', model_path = 'flashbang_model.pkl', train_test_split = 0.8, max_each_class = 1300):
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
        
        self.classes = [0, 1]
        self.label_dict = {'available': 0, 'unavailable': 1}

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
    def __init__(self, save_folder = f'flashbang_frames/', resolution = (1920, 1080), time_between_frames = 0.05, start_file_number = 0):
        # Save Location
        self.save_folder = save_folder
        self.file_number = start_file_number

        # Resolution
        self.x_res = resolution[0]
        self.y_res = resolution[1]

        # Time Between Frames
        self.time_between_frames = time_between_frames

        # Initialize screen capture to ammo location
        top_left = (round(self.y_res*135/160), round(self.x_res*134/160))
        width = round(self.x_res*140/160) - top_left[1]
        height = round(self.y_res*145/160) - top_left[0]
        self.monitor = {'top': top_left[0], 'left': top_left[1], 'width': width, 'height': height}

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
            # cv2.imshow('frame', frame)
            # cv2.waitKey(1)
            self.save_frame(frame)
            # Wait
            time.sleep(self.time_between_frames)
