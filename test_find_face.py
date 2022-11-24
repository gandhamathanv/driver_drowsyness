# import the necessary packages


import cv2

import winsound


face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_default.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

eye_cascade = cv2.CascadeClassifier('haar cascade files\haarcascade_eye_tree_eyeglasses.xml')
 
# mouth = cv2.CascadeClassifier('haar cascade files\mouth.xml')



video=cv2.VideoCapture(0)
#video=cv2.VideoCapture('https://tenor.com/view/smile-fake-drive-safe-driving-gif-13990142')

while(True):
    ret,frame=video.read()
    height,width = frame.shape[:2] 
    
    
    
    #faces=facedetect.detectMultiScale(frame,1.1,4)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #HISTOGRAM BY CHANGING ALPHA BETA IN SAME CONVERSION METHOD
    a=cv2.convertScaleAbs(gray,alpha=3.10,beta=5)
    
    
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,255,0) , 3 )
    
    
   
       
    eyes = eye_cascade.detectMultiScale(gray)
    for (ex, ey ,ew, eh) in eyes:
        cv2.rectangle(frame, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
   
    
#     mo = mouth.detectMultiScale(gray)
#     for (mx, my ,mw, mh) in mo:
#         cv2.rectangle(frame, (mx,my), (mx+mw, my+mh), (0, 255, 0), 2)
        
    
    cv2.imshow('frame',frame)
    #cv2.imshow('Gray',gray)
    #cv2.imshow('Histogram',a)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
