from mss import mss
import cv2
import threading
import numpy as np
import mouse
import pyautogui
import time
# SkLearn
import sklearn
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
import glob
import pickle
# matplotlib 3.3.1
from matplotlib import pyplot

class TrainModel:
    def __init__(self, data_path = 'character_identifier_data/', train_test_split = 0.8, max_each_class = 3000):
        self.model = None
        self.model_path = 'character_identifier.pkl'

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
        
        self.classes = [0, 1, 2]
        self.label_dict = {'tracer': 0, 'ashe': 1, 'none': 2}

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
        # Drop Alpha Channel
        img = img[:, :, :3]
        # Get Center third of image
        img = img[round(img.shape[0]/3):round(img.shape[0]*2/3), round(img.shape[1]/3):round(img.shape[1]*2/3)]
        # Convert to Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # Resize
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
        # MLP Classifier
        self.model = MLPClassifier(max_iter=7, verbose=1, activation='relu', learning_rate_init=0.01) # Will change learning_rate_init to 0.001 later
        # Fit Data
        self.model.fit(self.train_data_read, self.train_labels)
        # Save model
        self.save_model()
        # Print Accuracy
        plot_confusion_matrix(self.model, self.test_data_read, self.test_labels)
        pyplot.show()
        self.test(self.model, self.test_data_read, self.test_labels)

    def test(self, model, test_x_final, test_y_final):
        preds = model.predict(test_x_final)
        correct = 0
        incorrect = 0
        for pred, gt in zip(preds, test_y_final):
            if pred == gt: correct += 1
            else: incorrect += 1
        print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")
        pyplot.show()
    
    def save_model(self):
        pickle.dump(self.model, open(self.model_path, 'wb'))

    def load_model(self):
        self.model = pickle.load(open(self.model_path, 'rb'))


class GatherData:
    def __init__(self, save_folder = 'data', character_name = 'other', resolution = (1920, 1080), time_between_frames = 0.1, start_file_number = 0):
        # Member variables
        self.character_name = character_name
        self.save_folder = save_folder
        # Resolution
        self.x_res = resolution[0]
        self.y_res = resolution[1]

        # Time Between Frames
        self.time_between_frames = time_between_frames

        # BGR Separators for Red
        self.red_min = 180
        self.red_max = 255
        self.green_max = 100
        self.blue_max = 100

        # Initialize save location
        self.save_location = self.save_folder + '/' + self.character_name + '/'
        self.file_number = start_file_number

        # Initialize screen capture
        self.monitor = {'top': 0, 'left': 0, 'width': self.x_res, 'height': self.y_res}

        # Initialize Capture Thread
        self.capture_thread = threading.Thread(target=self.capture_loop).start()

    def get_frame(self):
        with mss() as sct:
            # Get Screen Frame
            frame = np.array(sct.grab(self.monitor))
        return frame

    def process_frame_for_save(self, frame):
        # Drop Alpha Channel
        frame = frame[:, :, :3]
        # Filtered Red Locations
        frame = cv2.inRange(frame, (0, 0, self.red_min), (self.blue_max, self.green_max, self.red_max))
        return frame

    def save_frame(self, frame):
        # Save Frame
        cv2.imwrite(f'{self.save_location}{self.character_name}{self.file_number}.png', frame)
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



# class CharacterIdentifier:
#     def __init__(self, model_path):
#         self.model = self.load_model()
#         self.label_dict_reverse = {0: 'tracer', 1: 'ashe', 2: 'none'}

#         # Model Path
#         self.model_path = 'character_identifier.pkl'

#         # BGR Separators for Red
#         self.red_min = 180
#         self.red_max = 255
#         self.green_max = 100
#         self.blue_max = 100

#         # Downsize Proportion
#         self.downsize_proportion = 0.3

#     # Apply Transformations before Network Input
#     def apply_transformation(self, img):
#         # Drop Alpha Channel
#         img = img[:, :, :3]
#         # Filtered Red Locations
#         img = cv2.inRange(img, (0, 0, self.red_min), (self.blue_max, self.green_max, self.red_max))
#         # Get Center third of image
#         img = img[round(img.shape[0]/3):round(img.shape[0]*2/3), round(img.shape[1]/3):round(img.shape[1]*2/3)]
#         # Resize for Network Input
#         img = img.reshape(-1)
#         # Return Processed Image for Network
#         return img

#     def predict(self, screenshot):
#         processed_image = self.apply_transformation(screenshot)
#         return self.label_dict_reverse[self.model.predict(processed_image.reshape(1, -1))[0]]

#     def load_model(self):
#         with open(self.model_path, 'rb') as f:
#             model = pickle.load(f)
#         return model