import cv2
from services.drowsiness.DrowsinessDetector import DrowsinessDetector
from services.headpose.HeadposeDetector import HeadposeDetector
from services.emotion.Engine.EmotionDetector import EmotionDetector


class VisualFocusMeter:
    def __init__(self):
        self.drowsiness_detector = DrowsinessDetector()
        self.headpose_detector = HeadposeDetector()
        self.emotion_detector = EmotionDetector()

        self.DROWSINESS_PERCENTAGE = 1 / 3
        self.HEADPOSE_PERCENTAGE = 1 / 3
        self.EMOTION_PERCENTAGE = 1 / 3

    def get_visual_focus(self, image):
        drowsiness = self.drowsiness_detector.get_drowsiness(image)
        unfocusness, image = self.headpose_detector.get_unfocus_headpose_percentage(
            image
        )
        emotion = self.emotion_detector.get_emotional_focus(image)

        drowsiness_cofficient = 0 if drowsiness else 1
        headpose_coffiecient = 1 - unfocusness if unfocusness is not None else 0
        emotion_coffiecient = 1 if emotion else 0

        focuss_amount = (
            drowsiness_cofficient * self.DROWSINESS_PERCENTAGE
            + headpose_coffiecient * self.HEADPOSE_PERCENTAGE
            + emotion_coffiecient * self.EMOTION_PERCENTAGE
        )

        return focuss_amount


focusmeter = VisualFocusMeter()

sample_image = cv2.imread("sample3.jpg")
result = focusmeter.get_visual_focus(sample_image)
print(result)
