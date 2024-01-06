import cv2
import mediapipe as mp
import numpy as np


class DrowsinessDetector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            min_tracking_confidence=0.6,
            min_detection_confidence=0.6,
        )

    def _get_right_eye_rect(self, landmarks, frame_width, frame_height):
        p1 = landmarks[386]
        p2 = landmarks[263]
        p3 = landmarks[374]
        p4 = landmarks[362]

        x1 = int(min(p1.x, p2.x, p3.x, p4.x) * frame_width)
        y1 = int(min(p1.y, p2.y, p3.y, p4.y) * frame_height)
        x2 = int(max(p1.x, p2.x, p3.x, p4.x) * frame_width)
        y2 = int(max(p1.y, p2.y, p3.y, p4.y) * frame_height)

        width = x2 - x1
        height = y2 - y1

        return np.asarray([x1, y1, width, height], np.int64)

    def _get_left_eye_rect(self, landmarks, frame_width, frame_height):
        p1 = landmarks[159]
        p2 = landmarks[133]
        p3 = landmarks[145]
        p4 = landmarks[33]

        x1 = int(min(p1.x, p2.x, p3.x, p4.x) * frame_width)
        y1 = int(min(p1.y, p2.y, p3.y, p4.y) * frame_height)
        x2 = int(max(p1.x, p2.x, p3.x, p4.x) * frame_width)
        y2 = int(max(p1.y, p2.y, p3.y, p4.y) * frame_height)

        width = x2 - x1
        height = y2 - y1

        return np.asarray([x1, y1, width, height], np.int64)

    def _get_mouth_rect(self, landmarks, frame_width, frame_height):
        p1 = landmarks[13]
        p2 = landmarks[308]
        p3 = landmarks[14]
        p4 = landmarks[78]

        x1 = int(min(p1.x, p2.x, p3.x, p4.x) * frame_width)
        y1 = int(min(p1.y, p2.y, p3.y, p4.y) * frame_height)
        x2 = int(max(p1.x, p2.x, p3.x, p4.x) * frame_width)
        y2 = int(max(p1.y, p2.y, p3.y, p4.y) * frame_height)

        width = x2 - x1
        height = y2 - y1

        return np.asarray([x1, y1, width, height], np.int64)

    def _calc_aspect_ratio(self, rect):
        x, y, w, h = rect
        return h / w

    def _mark_rect(self, image, rect):
        x, y, w, h = rect
        cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
        return image

    def _get_landmark(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        op = self.face_mesh.process(rgb)
        return op.multi_face_landmarks[0].landmark if op.multi_face_landmarks else None

    def get_drowsiness_parameters(self, image):
        height, width, _ = image.shape

        landmarks = self._get_landmark(image)

        if not landmarks:
            return None

        right_eye_rect = self._get_right_eye_rect(landmarks, width, height)
        left_eye_rect = self._get_left_eye_rect(landmarks, width, height)
        mouth_rect = self._get_mouth_rect(landmarks, width, height)

        right_eye_aspect_ratio = self._calc_aspect_ratio(right_eye_rect)
        left_eye_aspect_ratio = self._calc_aspect_ratio(left_eye_rect)
        mouth_aspect_ratio = self._calc_aspect_ratio(mouth_rect)

        EAR = (right_eye_aspect_ratio + left_eye_aspect_ratio) / 2
        YAR = mouth_aspect_ratio

        return EAR, YAR

    def get_drowsiness(self, image):
        EAR, YAR = self.get_drowsiness_parameters(image)
        if EAR < 0.2 and YAR > 0.4:
            return True
        return False

    def mark_image(self, image):
        height, width, _ = image.shape

        landmarks = self._get_landmark(image)
        if not landmarks:
            return image

        right_eye_rect = self._get_right_eye_rect(landmarks, width, height)
        left_eye_rect = self._get_left_eye_rect(landmarks, width, height)
        mouth_rect = self._get_mouth_rect(landmarks, width, height)

        image = self._mark_rect(image, right_eye_rect)
        image = self._mark_rect(image, left_eye_rect)
        image = self._mark_rect(image, mouth_rect)

        return image


# capture = cv2.VideoCapture(0)

# FRAME_WIDTH = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
# FRAME_HEIGHT = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))


# while True:
#     ret, frame = capture.read()
#     if ret:
#         detector = DrowsinessDetector()
#         parameters = detector.get_drowsiness_parameters(frame)
#         if parameters:
#             EAR, YAR = parameters
#             print(EAR, YAR)
#         frame = detector.mark_image(frame)
#         cv2.imshow("photo", frame)
#         key = cv2.waitKey(1)
#         if key == ord("q"):
#             break
