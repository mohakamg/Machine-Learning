from __future__ import division
import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import cos, sin

# Initialize the Camera to use. 0 defaults the 1st webcam available
cap = cv2.VideoCapture(0)

# Run In an infinite loop to get frames
while True:
    # Get the frame
    _, frame = cap.read()

    # Convert original image to Grey
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    min_color = np.array([150,150,0]) # Minimum amount of redness
    max_color = np.array([180,255,150]) # Maximum amount of redness

        # Mask the image pixles in the color range
    mask = cv2.inRange(hsv, min_color, max_color)

    res = cv2.bitwise_and(frame,frame, mask=mask)

    ######### FILTER NOISE AND SMOOTH
    # 2D Filter Smoothing
    # kernel = np.ones((15,15), np.float32)/255
    # smoothed  = cv2.filter2D(res, -1, kernel)

    # Gaussian Blur
    # blur = cv2.GaussianBlur(res, (15,15), 0)

    # Median Blur
    median = cv2.medianBlur(res, 15)

    # Bilateral Blur
    #bilateral = cv2.bilateralFilter(res, 15, 75 ,75)

    # Display the frames
    # cv2.imshow('res', res)
    # cv2.imshow('smoothed', smoothed)
    # cv2.imshow('blur', blur)

    # MORPHOLOGICAL TRANSFORM - Errosion and Dilation
    kernel  = np.ones((5,5), np.uint8)
    # erosion = cv2.erode(median, kernel, iterations=1)
    # dilation = cv2.dilate(erosion, kernel, iterations=1)
    opening = cv2.morphologyEx(median, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    cup_edges = cv2.Canny(closing, 100, 200)

    result = cv2.add(cup_edges, gray)

    # Show
    cv2.imshow('Image', result)


    # Exit out of the loop if 'q' pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Destroy the Windows
cv2.destroyAllWindows()
# Release the camera
cap.release()
