from __future__ import print_function

import numpy as py #importing packages
import math
import cv2
import pymongo

myclient = pymongo.MongoClient("mongodb://localhost:27017/") #Setting up MongoDB
mydb = myclient["mydatabase"]
mycol = mydb["customers"]

counter=0

bodyCascade = cv2.CascadeClassifier("/Users/kaushiknambi/Desktop/haarcascades_fullbody.xml") #Body Cascade

cv2.startWindowThread()
video_capture = cv2.VideoCapture(1)

ret, frame1 = video_capture.read()
frame1 = cv2.resize(frame1, (640,480))
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = py.zeros_like(frame1)
hsv[...,1] = 255
PeopleInDC=0

while True:
    # Capture frame-by-frame
    ret, frame2 = video_capture.read()
    frame2 = cv2.resize(frame2, (640, 480))
    next = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0) #Optical Flow to determine direction
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/py.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    flow2 = py.mean(flow, axis=0)
    flow3 = py.mean(flow2, axis=0)
    flow4 = py.mean(flow3, axis=0)
    print("Average Velocity",flow4)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('Optical Flow ',rgb) # Showing the frame in which optical flow is seen
    
    j=cv2.waitKey(30) & 0xff
    if j==27 :
        break
    elif j == ord('s'):
        cv2.imwrite('opticalfb.png',frame2) #Displaying the right frame
        cv2.imwrite('opticalhsv.png',rgb)
    prvs = next

    if flow4<-0.75 :
        PeopleInDC-=1
    elif flow4>0.75 :
        PeopleInDC+=1
    
    print("People in the DC :\n", PeopleInDC)

    if counter%120==0 :
        mydict = {"Number": PeopleInDC}
        x=mycol.insert_one(mydict)
        print(x)
        myquery = {"Number": PeopleInDC}
        mycol.delete_one(myquery)
    
    k = cv2.waitKey(1)
    body = bodyCascade.detectMultiScale(
       next, scaleFactor=1.0476258, minNeighbors=2, minSize=(5, 6),
        flags=cv2.CASCADE_SCALE_IMAGE)
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in body:
        cv2.rectangle(next, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Body Detection', next)
        
#When everything is done, release the capture
video_capture.release()

cv2.destroyAllWindows()
