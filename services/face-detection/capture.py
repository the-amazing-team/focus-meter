import cv2

webcam = cv2.VideoCapture(0)
stop = False

while not stop:
    ret, frame = webcam.read()
    if ret:
        cv2.imshow("photo", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            stop = True

webcam.release()
cv2.destroyAllWindows()
#import 
#make  a --> var =  cv2.webcapture(0)
#stop = False

#while not stop :
# ret, frame = webcam.read()
# if ret :
#cv2.imshow("photo", frame)
# ket = cv2.waitket(1)
#if key == ors('q):
#stop = true




