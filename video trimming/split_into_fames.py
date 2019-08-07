import cv2
import numpy as np
import os
from progressBar import printProgressBar

# Deubgging Variables
debugging = False
start = 100
end = 500


# Conf variables
video_name = '06_21_2019_02'


video_file = video_name + '.mp4'
currentFrame = 0
output_folder = 'staging_area/' + video_name
cap = cv2.VideoCapture(video_file)
fps = cap.get(cv2.CAP_PROP_FPS)
totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
success, frame = cap.read()

try:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
except OSError:
    print('Error: Could not create output_folder directory')

print("Please Wait...")
while(success):
    name = './' + output_folder + '/' + str('%06d' % currentFrame) + '.jpg'
    
    if debugging:
        if currentFrame >= start and currentFrame <= end and currentFrame % fps == 0:
            cv2.imwrite(name, frame)
    else:    
        if currentFrame % fps == 0:
            cv2.imwrite(name, frame)

    if debugging:
        if currentFrame >= end:
            break

    # Printing progress bar
    printProgressBar(currentFrame, totalFrames, prefix='Progress:', suffix='Progress', length=50)
    # for next iteration
    currentFrame += 1
    success, frame = cap.read()

# release the memory with cap variable
cap.release()
cv2.destroyAllWindows()

print("\nComplete")