# Creator: Gaurav Grover
# Date: 29 june 2019
# This script was create for testing out some python APIs before implementing it into the code

from time import sleep
import cv2

cap = cv2.VideoCapture("01_31_2019_01.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)

# length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(length)