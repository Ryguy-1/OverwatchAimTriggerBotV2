from screen import Window
from trainer import GatherData
from trainer import Train

# Sets up Main Window
def main():
    ui = Window(uses_trigger_bot=True, uses_ammo_analyzer=True, uses_flashbang_analyzer=True, time_between_frames_ms=1)
    
if __name__ == '__main__':
    main()