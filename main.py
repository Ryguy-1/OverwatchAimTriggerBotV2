from numpy import character
from screen import TriggerBot
from character_identifier import GatherData
from character_identifier import TrainModel
from character_identifier import CharacterIdentifier
import cv2
import time
from mss import mss
import numpy as np

def main():
    trigger_bot = TriggerBot(resolution=(1920, 1080), snap_square_length=2, fire_delay=0.20)
    
def get_frame():
    monitor = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
    with mss() as sct:
        # Get Screen Frame
        frame = np.array(sct.grab(monitor))
    return frame

if __name__ == '__main__':

    main()