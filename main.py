from screen import Window
from trainer import GatherData
from trainer import Train
from character_identifier import GatherData
from character_identifier import TrainModel

# Sets up Main Window
def main():
    Window(uses_trigger_bot=True, uses_ammo_analyzer=True, uses_flashbang_analyzer=True, uses_character_identifier = True, time_between_frames_ms=1, ui_dimensions=(1920, 1080))
    
if __name__ == '__main__':
    # gather_data = GatherData(save_folder="character_identifier_data", character_name="none")
    # train_model = TrainModel(train_test_split=0.9)
    main()