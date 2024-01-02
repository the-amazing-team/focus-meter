import cv2
import mediapipe 

capt = cv2.VideoCapture(0)
stop  = False

while not stop :
    ret , frame = capt.read()
    if ret :
        cv2.imshow("photo", frame)
        key_to_stop = cv2.waitKey()
        if key_to_stop == ord('q'):
            stop = True

capt.release()
cv2.destroyAllWindows()    