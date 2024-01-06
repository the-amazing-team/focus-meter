import cv2
from services.drowsiness.DrowsinessDetector import DrowsinessDetector
from services.headpose.HeadposeDetector import HeadposeDetector

# from services.headpose import HeadposeDetector
from services.emotion.Engine.EmotionDetector import EmotionDetector

drowsiness_detector = DrowsinessDetector()
headpose_detector = HeadposeDetector()
emotion_detector = EmotionDetector()

sample_image = cv2.imread("sample7.jpg")

# cv2.imshow("photo", sample_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# result = drowsiness_detector.get_drowsiness(sample_image)
# unfocusness, image = headpose_detector.get_unfocus_headpose_percentage(sample_image)
emotion = emotion_detector.get_emotional_focus(sample_image)
print(emotion)