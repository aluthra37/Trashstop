import numpy as np
import cv2

# Opening VideoCapture
cam = cv2.VideoCapture(0)
cv2.namedWindow('Capture')
imgCount=0

# Loop Runs if Capture Initialized
while(True):

# ret Checks Frame Return
    ret, frame = cam.read()

# Shows the Original Frame
    cv2.imshow('Capture', frame)

    i=cv2.waitKey(1)
# ' ' Key Initializes Frame Grab
    if i==32:
        imgName = 'frame_{}.png'.format(imgCount)
        cv2.imwrite('/Users/ruthvikpedibhotla/Documents/Python/garbage_photos/{}'.format(imgName), frame)
        print('{} written'.format(imgName))
        imgCount += 1
# esc Key Quits
    elif i==27:
        break
# Close the window
cam.release()

# De-allocate Memory Useage
cv2.destroyAllWindows()
