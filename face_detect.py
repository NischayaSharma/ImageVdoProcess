import cv2
import sys

# Taking input of the image from the terminal/shell
imagePath = sys.argv[1]
# Cascade path, It will be a .xml file
# Look in the folder you installed opencv
cascPath = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascPath)

# reading the Image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# The code which detects the Faces, play around with scale factor and see
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30, 30)
)
print("Found {0} faces!".format(len(faces)))

# Marking rectangle around the faces on the image
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imshow("Faces found", image)
cv2.waitKey(2000)
cv2.destroyAllWindows()
