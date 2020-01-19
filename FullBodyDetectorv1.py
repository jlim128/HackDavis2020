import cv2
import numpy as py

bodyCascade = cv2.CascadeClassifier("/Users/kaushiknambi/Desktop/haarcascades_fullbody.xml")

cv2.startWindowThread()
video_capture = cv2.VideoCapture(1)


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    k = cv2.waitKey(1)
    body = bodyCascade.detectMultiScale(
       gray, scaleFactor=1.0476258, minNeighbors=2, minSize=(5, 6),
        flags=cv2.CASCADE_SCALE_IMAGE)
    

    # Draw a rectangle around the faces
    for (x, y, w, h) in body:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Body Detection', frame)
        

#When everything is done, release the capture
video_capture.release()

cv2.destroyAllWindows()
