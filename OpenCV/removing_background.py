import numpy as np
import cv2

img1 = cv2.imread('3D-Matplotlib.png')
img2 = cv2.imread('mainlogo.png')


rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]

img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)

mask_inv = cv2.bitwise_not(mask)
img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
img_fg = cv2.bitwise_and(img2,img2, mask=mask)
# img1[0:rows, 0:cols] = img
img = cv2.add(img_bg, img_fg)
img1[0:rows, 0:cols] = img
#
# add = img1 + img2
# add2 = cv2.add(img1,img2)
#
# weighted = cv2.addWeighted(img1,0.6, img2, 0.4, 0.5)
cv2.imshow("Add",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
