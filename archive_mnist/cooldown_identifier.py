import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
import cv2
import imutils



class TrainModel:
    def __init__(self, test_size=0.2, random_state=0, model_name = "mnist.pkl"):
        # Model Name
        self.model_name = model_name
        # Random State
        self.random_state = random_state
        # Test Size
        self.test_size = test_size
        # Load Data
        qmnist = self.unpickle("MNIST-120k")
        self.X = np.array(qmnist['data'])
        self.y = np.array(qmnist['labels'])
        # Reshape Data to -1, 784
        self.X = self.X.reshape((-1, 784))
        # Print Shape of Data
        print(f'X Shape: {self.X.shape}')
        print(f'y Shape: {self.y.shape}')
        # Split Data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state, shuffle=True)
        # SVM Classifier
        self.clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=10000, verbose=1)
        # Train Model
        print(f'X Shape: {self.X_train.shape}')
        print(f'y Shape: {self.y_train.shape}')
        # Show first 10 images
        for i in range(10):
            plt.imshow(self.X_train[i].reshape(28, 28), cmap='gray')
            plt.show()
        self.clf.fit(self.X_train, self.y_train)
        # Predict
        self.y_pred = self.clf.predict(self.X_test)
        # Accuracy
        self.accuracy = metrics.accuracy_score(self.y_test, self.y_pred)
        print(self.accuracy)
        # Confusion Matrix
        metrics.plot_confusion_matrix(self.clf, self.X_test, self.y_test)
        plt.show()
        # Save Model
        self.save_model()

    def save_model(self):
        # Save Model
        pickle.dump(self.clf, open(self.model_name, "wb"))

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

class UseModel:
    def __init__(self, model_name = "mnist.pkl", resolution = (1920, 1080)):

        # Resolution
        self.x_res = resolution[0]
        self.y_res = resolution[1]

        # Model Name
        self.model_name = model_name
        # Load Model
        self.clf = pickle.load(open(self.model_name, "rb"))

        # Threshold Values
        self.threshold_low = 160
        self.threshold_high = 255

    def crop_amunition(self, image):
        # Crop Amunition
        image = image[round(self.y_res*263/320):round(self.y_res*137/160), round(self.x_res*297/320):round(self.x_res*151/160)]
        return image

    def apply_transforms(self, image):
        # Crop Amunition
        image = self.crop_amunition(image)
        # Convert to Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Threshold
        image = cv2.threshold(image, self.threshold_low, self.threshold_high, cv2.THRESH_BINARY)[1]
        # Blur
        image = cv2.GaussianBlur(image, (3, 3), 0)
        # Show Before Resize
        plt.imshow(image, cmap='gray')
        plt.show()
        # Resize
        image = cv2.resize(image, (28, 28))
        # Show With matplotlib
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        plt.show()
        # Reshape
        image = image.reshape(1, -1)
        # Return
        return image

    def test_single(self, test_image):
        # Apply Transforms
        test_image = self.apply_transforms(test_image)
        # Predict
        prediction = self.clf.predict(test_image)
        # Print
        print(f'Prediction: {prediction[0]}')
        # Return
        return prediction
