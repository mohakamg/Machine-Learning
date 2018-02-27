import cv2
import numpy as np

img = cv2.imread('bookpage.jpg')
imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
retval, threshold = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)
retval2, threshold2 = cv2.threshold(imggray, 12, 255, cv2.THRESH_BINARY)
gaus_threshold = cv2.adaptiveThreshold(imggray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 115,1)
cv2.imshow('original', imggray)
cv2.imshow('threshold', threshold)
cv2.imshow('threshold2', threshold2)
cv2.imshow('gaur threshold', gaus_threshold)

cv2.waitKey(0)
cv2.destroyAllWindows()
