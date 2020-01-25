import cv2
import pandas as pd
import imutils

image = cv2.imread("media/bill.jpg")
cv2.imshow("BILL", image)
cv2.waitKey(0)

(h,w,d) = image.shape
print("width: {}, height: {}, depth: {}".format(w,h,d))

ratio = 300/w
resized = cv2.resize(image,(300,int(h*ratio)))
cv2.imshow("new", resized)
cv2.waitKey(0)

grayimg = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
grayimg = cv2.GaussianBlur(grayimg, (5, 5), 0)
edged = cv2.Canny(grayimg, 15, 100)
cv2.imshow("Image", resized)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
 
# loop over the contours
# for c in cnts:
# 	# approximate the contour
# 	peri = cv2.arcLength(c, True)
# 	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 
# 	# if our approximated contour has four points, then we
# 	# can assume that we have found our screen
# 	if len(approx) == 4:
# 		screenCnt = approx
# 		break
 
# show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [cnts], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()