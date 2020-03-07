import cv2
import sys

# Original Image
image = cv2.imread(sys.argv[1])
# Original Image in Grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# Blurring the GrayScale Image to get rid of shadows
blur = cv2.GaussianBlur(gray,(7,7),0)

# Using Canny's Algo for Edge Detection
#  Low Threshold (10,30)
cannyLow = cv2.Canny(image, 10,30)
#  Hight Threshold (50,150)
cannyHigh = cv2.Canny(image, 250,250)

# Object Counting from the above detected edges
# The first option is the output of the canny edge detector. 
# RETR_EXTERNAL tells OpenCv to only find the outermost edges (as you can find contours within contours)
# The second arguments tells OpenCv to use the simple approximation
# Hierarchy is used if you have many contours embedded within others
contours, hierarchy= cv2.findContours(cannyHigh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Displaying all Images
print("Number of objects found = ", len(contours))
cv2.imshow("Original Image", image)
cv2.imshow("Gray Image", gray)
cv2.imshow("Blurred Image", blur)
cv2.imshow("Canny low Threshold", cannyLow)
cv2.imshow("Canny high Threshold", cannyHigh)
# Highlighting the found Objects in image
cv2.drawContours(image, contours, -1, (0,255,0), 2)
cv2.imshow("objects Found", image)
cv2.waitKey(0)