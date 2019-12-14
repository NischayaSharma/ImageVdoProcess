import cv2
import sys
import numpy as np

# If a video file is not given from the terminal VideoCapture(0) points to default Webcam of your laptop
# Else take the given Video file as input
if len(sys.argv) < 2:
    video_capture = cv2.VideoCapture(0)
else:
    video_capture = cv2.VideoCapture(sys.argv[1])

ret, prev_frame = video_capture.read()
ret, curr_frame = video_capture.read()
gray = cv2.cvtColor(curr_frame,cv2.COLOR_BGR2GRAY)
while True:
    prev_frame = curr_frame
    ret, curr_frame = video_capture.read()
    if not ret:
        break
    gray = cv2.cvtColor(curr_frame,cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_frame, curr_frame)
    # print(np.mean(diff))
    if np.mean(diff) >= 5:
        print("Motion detected.")
    cv2.imshow('Video',diff)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
video_capture.release()
cv2.destroyAllWindows()