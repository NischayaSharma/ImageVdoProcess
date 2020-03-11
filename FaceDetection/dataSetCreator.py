import numpy as np
import cv2
import sqlite3
import os
import shutil

cap = cv2.VideoCapture(0)
detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def convertToBinaryData(filename):
    with open(filename, 'rb') as file:
        blobData = file.read()
    return blobData

def insert(Name,img):
    conn=sqlite3.connect("FaceBase.db")
    cmd="INSERT INTO people(name,face) Values(?,?)"
    conn.execute(cmd,(Name,img))
    conn.commit()
    conn.close()

def cleanup():
    shutil.rmtree("dataSet")

name=str(input('enter your name'))
sampleNum=0

while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        sampleNum=sampleNum+1
        if sampleNum == 1:
            os.mkdir("dataSet")
        cv2.imwrite("dataSet/User."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        face = convertToBinaryData("dataSet/User."+str(sampleNum)+".jpg")
        insert(name,face)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.waitKey(100)

    cv2.imshow('Face',img)
    cv2.waitKey(1)
    if(sampleNum>300):
        break

cap.release()
cv2.destroyAllWindows()
cleanup()