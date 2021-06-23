import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier(r'C:\Users\ADMIN\Desktop\haar_face.xml')

people = ['Hai', 'Trong Dat', 'stranger', 'Tri','Hieu']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

capture = cv.VideoCapture(0)
while True:
    IsTrue, frame = capture.read()

    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    #Tìm số khuôn mặt phát hiện được trong khung hình
    faces = haar_cascade.detectMultiScale(gray,1.1,3)

    for (x,y,w,h) in faces:
        face_roi = gray[y:y+h,x:x+h]

        label, confidence = face_recognizer.predict(face_roi)
        print(f'Label = {people[label]} with a confidence of {confidence}')

        cv.putText(frame,str(people[label]),(200,200),cv.FONT_HERSHEY_PLAIN,1.0,(0,191,255), 2)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,191,255),2)

    cv.imshow('Video',frame)

    if cv.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()

