import cv2
import sys

# Cascade path, It will be a .xml file
# Look in the folder you installed opencv
cascPath = "/home/nischaya/OpenCV/data/haarcascades/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascPath)

# If a video file is not given from the terminal VideoCapture(0) points to default Webcam of your laptop
# Else take the given Video file as input
if len(sys.argv) < 2:
    video_capture = cv2.VideoCapture(0)
else:
    video_capture = cv2.VideoCapture(sys.argv[1])

# Infinite loop to read the video frame by frame
while True:
    # Capturing every frame as an Image
    ret, image = video_capture.read()
    # Checking thhe next frame exists, will be usefull when the given Video file ends
    if not ret:
        break
    # Same code for face detection used in Images
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Commented because gives a bulky realtime output of the number of faces detected
    # print("The number of faces found = ", len(faces))

    # Drawing a green rectangle around the face
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+h, y+h), (0, 255, 0), 2)
    cv2.imshow("Faces found", image)

    # waiting for the Key "q" to be pressed to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()