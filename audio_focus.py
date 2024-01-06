from services.offensive_language_detector.OffensiveLanguageDetector import (
    OffensiveLanguageDetector,
)
from services.vocal_emotion.VocalEmotionDetection import VocalEmotionDetection
from services.speech_to_text.SpeechToText import SpeechToText


class AudioFocusMeter:
    def __init__(self):
        self.offensive_detector = OffensiveLanguageDetector()
        self.vocal_detector = VocalEmotionDetection()
        self.recognizer = SpeechToText()

        self.OFFENSIVE_PERCENTAGE = 1 / 2
        self.VOCAL_EMOTION_COFFIECIENT = 1 / 2

    def check_focus(self, audio_path):
        recognized_text = self.recognizer.recognize(audio_path)
        offensive_result = self.offensive_detector.detect(recognized_text)
        offensive_coffiecient = 1 - offensive_result["offensive"]
        vocal_emotion_coffiecient = (
            1 if self.vocal_detector.check_focus_wavefile(audio_path) else 0
        )
        audio_focus = (
            offensive_coffiecient * self.OFFENSIVE_PERCENTAGE
            + vocal_emotion_coffiecient * self.VOCAL_EMOTION_COFFIECIENT
        )
        return audio_focus


AUDIO_PATH = "sample.wav"

audio_focus_detector = AudioFocusMeter()
result = audio_focus_detector.check_focus(AUDIO_PATH)

print(result)
