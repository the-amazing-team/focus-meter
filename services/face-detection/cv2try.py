import cv2
import mediapipe as mp 

web = cv2.VideoCapture(0)
stop  = False
while not stop :
    ret, frame  =  web.read()
    if ret:
        cv2.imshow("photo", frame)
        key_to_end =  cv2.waitKey(500)
        if key_to_end == ord('q'):
            break

web.release()
cv2.destroyAllWindows()        

