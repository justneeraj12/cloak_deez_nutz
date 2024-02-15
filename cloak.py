# Import Libraries
import numpy as np
import cv2
import time

# To use webcam enter 0 and to enter the video path in double quotes
cap = cv2.VideoCapture(0)

time.sleep(3)

background = 0

# Capturing the background
for i in range(60):
    ret, background = cap.read()

background = np.flip(background, axis=1)

while(cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break
    img = np.flip(img, axis=1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Adjusting HSV range for blue color detection
    lower_blue = np.array([101, 50, 38])
    upper_blue = np.array([110, 255, 255])
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_blue = np.array([101, 50, 38])
    upper_blue = np.array([110, 255, 255])
    mask2 = cv2.inRange(hsv, lower_blue, upper_blue)

    mask1 = mask1 + mask2

    # Morphological operations
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8), iterations=1)
    mask2 = cv2.bitwise_not(mask1)

    # Generating final output
    res1 = cv2.bitwise_and(background, background, mask=mask1)
    res2 = cv2.bitwise_and(img, img, mask=mask2)
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    cv2.imshow('Invisible Cloak', final_output)
    k = cv2.waitKey(10)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
