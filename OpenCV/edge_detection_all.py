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

    # Convert original image to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(grey, (15,15), 0)
    #laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    #sobelx = cv2.Sobel(blur, cv2.CV_64F,1,0,ksize = 5)
    #sobely = cv2.Sobel(frame, cv2.CV_64F,0,1,ksize = 5)

    cv2.imshow('original', frame)
    # cv2.imshow('laplacian', laplacian)
    # cv2.imshow('sobelx', sobelx)
    # cv2.imshow('sobely', sobely)

    edges = cv2.Canny(grey, 100, 200)
    res = cv2.add(grey, edges)
    cv2.imshow('edges', res)

    # Exit out of the loop if 'q' pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Destroy the Windows
cv2.destroyAllWindows()
# Release the camera
cap.release()
