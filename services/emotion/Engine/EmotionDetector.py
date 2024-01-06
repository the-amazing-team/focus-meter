import cv2
from fer import FER
from services.emotion.Engine.FaceDetector import FaceDetector


class EmotionDetector(FaceDetector):
    def __init__(self):
        self.emotion_detector = FER(mtcnn=True)

    def detect_emotion(self, image):
        emotion_array = self.emotion_detector.detect_emotions(image)
        emotion_array = emotion_array[0]["emotions"] if len(emotion_array) else None
        max_emotion = (
            max(emotion_array, key=emotion_array.get)
            if emotion_array is not None
            else None
        )
        return max_emotion if max_emotion is not None else "neutral"

    def label_frame_with_emotion(self, image):
        emotion = self.detect_emotion(image)
        emotion_image = self.get_emotion_image(emotion)

        labelled_image = self.overlay_face_box_indicator(
            image=image,
            emotion_image=emotion_image,
            label_one=emotion,
        )

        return labelled_image

    def get_emotion_image(self, emotion_name):
        return cv2.imread(f"Emojis/{emotion_name}.png")

    def get_emotional_focus(self, image):
        emotion = self.detect_emotion(image)

        if emotion == "neutral" or emotion == "sad" or emotion == "fear":
            return True

        return False
