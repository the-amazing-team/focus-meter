import cv2
from services.headpose.network.network import Network
from services.headpose.utils import load_snapshot
from torchvision import transforms
import numpy as np
import torch
import time
from PIL import Image
from services.headpose.utils.camera_normalize import drawAxis
import math


class HeadposeDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            "./services/headpose/lbpcascade_frontalface_improved.xml"
        )
        self.pose_estimator = Network(bin_train=False)
        load_snapshot(self.pose_estimator, "./services/headpose/models/model-b66.pkl")
        self.pose_estimator = self.pose_estimator.eval()
        self.transform_test = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _scale_bbox(self, bbox, scale):
        w = max(bbox[2], bbox[3]) * scale
        x = max(bbox[0] + bbox[2] / 2 - w / 2, 0)
        y = max(bbox[1] + bbox[3] / 2 - w / 2, 0)
        return np.asarray([x, y, w, w], np.int64)

    def _detect_faces(self, image):
        count = 0
        last_faces = None

        if count % 5 == 0:
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_img, 1.2)
            if len(faces) == 0 and (last_faces is not None):
                faces = last_faces
            last_faces = faces

        return faces

    def _get_face_tensors_and_images(self, image, faces):
        face_images = []
        face_tensors = []

        for i, bbox in enumerate(faces):
            x, y, w, h = self._scale_bbox(bbox, 1.5)
            image = cv2.rectangle(
                image, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2
            )
            face_img = image[y : y + h, x : x + w]
            face_images.append(face_img)
            pil_img = Image.fromarray(
                cv2.cvtColor(cv2.resize(face_img, (224, 224)), cv2.COLOR_BGR2RGB)
            )
            face_tensors.append(self.transform_test(pil_img)[None])

        return face_tensors, face_images

    def _get_headpose(self, face_tensors, face_images):
        headposes = []

        with torch.no_grad():
            start = time.time()
            face_tensors = torch.cat(face_tensors, dim=0)
            roll, yaw, pitch = self.pose_estimator(face_tensors)
            print(
                "inference time: %.3f ms/face"
                % ((time.time() - start) / len(roll) * 1000)
            )
            for img, r, y, p in zip(face_images, roll, yaw, pitch):
                headpose = [r, y, p]
                headposes.append(headpose)

        return headposes

    def _mark_headposes(self, face_images, headposes):
        for img, headpose in zip(face_images, headposes):
            drawAxis(img, headpose, size=50)

    def detect_headpose(self, image):
        faces = self._detect_faces(image)
        if len(faces) == 0:
            return None, image

        face_tensors, face_images = self._get_face_tensors_and_images(image, faces)
        headposes = self._get_headpose(face_tensors, face_images)
        # restricting headpose detection to one person
        headposes = [headposes[0]] if len(headposes) > 0 else []
        self._mark_headposes(face_images, headposes)
        headposes = [headpose.item() for headpose in headposes[0]]
        headpose = headposes[0] if len(headposes) > 0 else None
        roll = headposes[0] if len(headposes) > 0 else None
        yaw = headposes[1] if len(headposes) > 0 else None
        pitch = headposes[2] if len(headposes) > 0 else None
        distance_from_center = math.sqrt(pow(yaw, 2) + pow(pitch, 2))
        return distance_from_center, image

    def get_unfocus_headpose_percentage(self, image):
        MAX_DEG = 25
        distance, image = self.detect_headpose(image)
        if distance is None:
            return None, image

        unfocus_degree = min(distance / MAX_DEG, 1)
        return unfocus_degree, image


# capture = cv2.VideoCapture(0)

# while True:

#     frame = cv2.imread("sample.png")
#     headpose_detector = HeadPoseDetector()

#     headposes, marked_image = headpose_detector.detect_headpose(frame)
#     print(headposes)

#     cv2.imshow("frame", marked_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
