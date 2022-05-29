import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import streamlit as st
import pandas as pd

st.title("Face Recognition System")
st.header('Attendance Monitoring')
run = st.checkbox('Click to take attendance')
FRAME_WINDOW = st.image([])

path = 'attendance'
images = []
names = []
list1 = os.listdir(path)
print(list1)

for i in list1:
    cur = cv2.imread(f'{path}/{i}')
    images.append(cur)
    names.append(os.path.splitext(i)[0])

def find_encoding(images):
    elist = []
    for j in images:
        j = cv2.cvtColor(j, cv2.COLOR_BGR2RGB)
        encodeimg = face_recognition.face_encodings(j)[0]
        elist.append(encodeimg)
    return elist

def Attendance_check(name):
    with open('Attendance.csv', 'r+') as f:
        datalist = f.readlines()
        print(datalist)
        namelist = []
        for l in datalist:
            entry = l.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            time1 = datetime.now()
            date1 = time1.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{date1}')

elistknown = find_encoding(images)
print('Checking for attendance. Please wait...')

cap = cv2.VideoCapture(0)

while run:
    success, j = cap.read()
    faces = cv2.resize(j, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facecurr = face_recognition.face_locations(faces)
    encodecurr = face_recognition.face_encodings(faces, facecurr)

    for enFace, faceLoc in zip(encodecurr, facecurr):
        matches = face_recognition.compare_faces(elistknown, enFace)
        dist = face_recognition.face_distance(elistknown, enFace)
        matchindex = np.argmin(dist)

        if matches[matchindex]:
            name = names[matchindex].upper()
            y1, x2, y2, x1 = faceLoc #top,left,bottom,right
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(j, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(j, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(j, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            Attendance_check(name)

    FRAME_WINDOW.image(j)
    #cv2.imshow('Real time image', j)
    #cv2.waitKey(0)
else:
    st.write('Attendance Done!')
    a1 = st.checkbox('Check Attendance')
    csv_file = 'Attendance.csv'
    df = pd.read_csv('D:/PycharmProjects/face_recognition1/Attendance.csv')

    while a1:
        st.write(df)
        break


