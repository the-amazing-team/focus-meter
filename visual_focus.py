import cv2
from services.drowsiness.DrowsinessDetector import DrowsinessDetector

# from services.headpose import HeadposeDetector
# from services.emotion.Engine import EmotionDetector

drowsiness_detector = DrowsinessDetector()
# headpose_detector = HeadposeDetector()
# emotion_detector = EmotionDetector()

sample_image = cv2.imread("sample3.jpg")
result = drowsiness_detector.get_drowsiness(sample_image)
print(result)
