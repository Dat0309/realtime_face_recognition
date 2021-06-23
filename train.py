import os
import cv2 as cv
import numpy as np

people = ['Hai', 'Trong Dat', 'stranger', 'Tri','Hieu']
DIR = r'C:\Users\ADMIN\Desktop\face'

haar_cascade = cv.CascadeClassifier(r'C:\Users\ADMIN\Desktop\haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_arr = cv.imread(img_path)

            gray = cv.cvtColor(img_arr,cv.COLOR_BGR2GRAY)

            face_rect = haar_cascade.detectMultiScale(gray,1.1,3)

            for (x,y,w,h) in face_rect:
                face_roi = gray[y:y+h,x:x+w]
                features.append(face_roi)
                labels.append(label)

create_train()
print("Training done .....................")

features  =np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')