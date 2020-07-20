import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

recogniser = cv2.face.LBPHFaceRecognizer_create()
recogniser.read('trainer.yml')

labels = {}
with open('labels.pickle', 'rb') as f:
    original_labels = pickle.load(f)
    labels = {v:k for k,v in labels.items()}


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
        if conf >= 45:
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            colour = (255,255,255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, colour, stroke, cv2.LINE_AA)
        
        img_item = 'my-image.png'
        cv2.imwrite(img_item, roi_grey)


        colour = (0,0,255)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), colour, stroke)
        eyes = eye_cascade.detectMultiScale(roi_grey)
        smile = smile_cascade.detectMultiScale(roi_grey)

        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_colour,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
        for(sx,sy,sw,sh) in smile:
            cv2.rectangle(roi_colour,(sx,sy),(sx+sw,sy+sh),(255,0,0),2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()