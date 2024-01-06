import cv2
import mediapipe as mp 
import numpy as np 

web = cv2.VideoCapture(0)

facemesh = mp.solutions.face_mesh #macemesh contaning FACEMESH
face =facemesh.FaceMesh(static_image_mode = True , min_tracking_confidence= 0.6, min_detection_confidence=0.6) #cls FACEMESH from 
draw = mp.solutions.drawing_utils





while True:


    _,frm  = web.read()
    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

    op = face.process( rgb)
    #print(op.multi_face_landmarks)
    if op.multi_face_landmarks:
       for i in op.multi_face_landmarks:
          draw.draw_landmarks(frm, i,  landmark_drawing_spec = draw.DrawingSpec( circle_radius=1, color = (0,255,255)) )
    cv2.imshow("window", frm)

    if cv2.waitKey(1) == 27:
     web.release()
     cv2.destroyAllWindows()
     break
