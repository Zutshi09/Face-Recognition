#RECONGNIZES FACE AND CREATES A DATASET FOR THE RECOGNIZED FACES


import cv2
import os

haar_file = "E:\Himanshu\Projects\Face Recoginition Project\Face detection source code\haarcascade_frontalface_default.xml"
datasets = "E:\Himanshu\Projects\Face Recoginition Project\Face detection source code\dataset"
sub_data = "E:\Himanshu\Projects\Face Recoginition Project\Face detection source code\dataset\Krishna"


path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)

(width, height) = (130, 100)

face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

count = 1
while count <= 50:
    print(count)
    ret, frame = webcam.read()
    if not ret:
        print("Error: failed to capture frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = gray[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite(f'{path}/{count}.png', face_resize)
        count += 1

    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
