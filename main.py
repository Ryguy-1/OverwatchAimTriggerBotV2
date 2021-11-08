from screen import TriggerBot, Window
import cv2
import time
from mss import mss
import numpy as np


def main():
    ui = Window(uses_trigger_bot=True, uses_ammo_analyzer=True, time_between_frames_ms=1)
    
if __name__ == '__main__':
    main()