import cv2
frame_capture =  cv2.VideoCapture(0)

currrent_frame  = 0

stop = False

while not stop :
    ret_frame, frame  =  frame_capture.read()
    if ret_frame == True:
        #cv2.imshow("photo", frame)
        cv2.imwrite("frame"+ str(currrent_frame)+'.jpg', frame)
        currrent_frame+=1
        #key_to_wait = cv2.waitKey(50)
        #if key_to_wait == ord('Q'):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

frame_capture.release()
cv2.destroyAllWindows()        
