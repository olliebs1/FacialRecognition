import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recogniser = cv2.face.LBPHFaceRecognizer_create()
recogniser.read('trainer.yml')


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, scaleFactor=1.5, minNeighbors=5)

    for(x, y, w, h) in faces:
        roi_grey = grey[y:y+h, x:x+w]
        roi_colour = frame[y:y+h, x:x+w]

        #recogniser
        id_, conf = recogniser.predict(roi_grey)
        if conf >= 45 and conf <= 85:
            print(id_)
        
        img_item = 'my-image.png'
        cv2.imwrite(img_item, roi_grey)


        colour = (0,0,255)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), colour, stroke)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()