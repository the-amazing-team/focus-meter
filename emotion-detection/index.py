import cv2
from Engine.EmotionDetector import EmotionDetector

emotion_detector = EmotionDetector()

video = cv2.VideoCapture(0)
cv2.namedWindow("LiveDetection")

# Creating Scale Factor Trackbar for Face Detector
cv2.createTrackbar(
    "(scaleFactor - 1) * 10", 
    "LiveDetection", 
    0, 20, 
    lambda scaleFactor_x10:
        emotion_detector.update_face_rectangle_parameteres(scaleFactor = ((scaleFactor_x10 / 10) + 1))
)

# Creating Minimum Neighbor Trackbar for Face Detector
cv2.createTrackbar(
    "minNeighbors - 1", 
    "LiveDetection", 
    0, 20, 
    lambda minNeighbors:
        emotion_detector.update_face_rectangle_parameteres(minNeighbors = (minNeighbors + 1))
)

while True:
    ret, frame = video.read()
    emotion = emotion_detector.detect_emotion(frame)
    print(emotion)
    emotion_image = emotion_detector.get_emotion_image(emotion)
    cv2.imshow("LiveDetection", emotion_image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()