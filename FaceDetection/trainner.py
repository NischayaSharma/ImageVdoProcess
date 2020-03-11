import os
import cv2
import numpy as np
from PIL import Image
import sqlite3
import shutil
recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataSet'

def writeTofile(data, filename):
    with open(filename, 'wb+') as file:
        file.write(data)
    print("Stored blob data into: ", filename, "\n")

def getData():
    sqliteConnection = sqlite3.connect('FaceBase.db')
    cursor = sqliteConnection.cursor()

    sql_fetch_blob_query = """SELECT * from People"""
    cursor.execute(sql_fetch_blob_query)
    record = cursor.fetchall()
    i=0
    for row in record:
        name  = row[1]
        photo = row[2]
        if i==0:
            os.mkdir("dataSet")
        photoPath = "dataSet/" + name + "."+str(i)+".jpg"
        writeTofile(photo, photoPath)
        i+=1

def cleanup():
    shutil.rmtree("dataSet")

def getImagesWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]

    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L');
        faceNp=np.array(faceImg,'uint8')
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("trainning",faceNp)
        cv2.waitKey(10)
    return IDs, faces

getData()
Ids,faces=getImagesWithID(path)
recognizer.train(faces,np.array(Ids))
recognizer.save('trainningData.yml')
cv2.destroyAllWindows()
cleanup()