from audio_focus import AudioFocusMeter
from visual_focus import VisualFocusMeter
import cv2


class FocusMeter:
    def __init__(self):
        self.audio_focusmeter = AudioFocusMeter()
        self.visual_focusmeter = VisualFocusMeter()

        self.audio_focus_sum = None
        self.visual_focus_count = None

        self.audio_focus_count = 0
        self.visual_focus_count = 0

        self.AUDIO_FOCUS_COFFECIENT = 1 / 2
        self.VISUAL_FOCUS_COFFECIENT = 1 / 2

    def reset_average(self):
        self.audio_focus_sum = None
        self.visual_focus_count = None
        self.audio_focus_count = 0
        self.visual_focus_count = 0

    def add_visual_feed(self, image):
        focus = self.visual_focusmeter.get_visual_focus(image)
        if self.visual_focus_sum == None:
            self.visual_focus_sum = focus
        else:
            self.visual_focus_sum += focus
        self.visual_focus_count += 1

    def add_audio_feed(self, audio_path):
        focus = self.audio_focusmeter.check_focus(audio_path)
        if self.audio_focus_sum == None:
            self.audio_focus_sum = focus
        else:
            self.audio_focus_sum += focus
        self.audio_focus_count += 1

    def calc_focus(self):
        average_visual_focus = self.audio_focus_sum / self.audio_focus_count
        average_audio_focus = self.audio_focus_sum / self.audio_focus_count
        net_focus = (
            average_audio_focus * self.AUDIO_FOCUS_COFFECIENT
            + average_visual_focus * self.VISUAL_FOCUS_COFFECIENT
        )
        return net_focus


focus_meter = FocusMeter()

AUDIO_PATH = "sample.wav"
IMAGE_PATH = "sample.png"

sample_image = cv2.imread(IMAGE_PATH)

focus_meter.add_visual_feed(sample_image)
focus_meter.add_audio_feed(AUDIO_PATH)

focus = focus_meter.calc_focus()
print(focus)
