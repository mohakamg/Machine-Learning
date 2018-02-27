import cv2
import numpy as np

# Import the Testing the image
img_bgr = cv2.imread('machine.jpg')
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Import the training Data
template = cv2.imread('port.jpg',0)
print(template.shape)
width, height = template.shape[::-1]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.5
loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_bgr, pt, (pt[0]+ , pt[1]+height), (0,255,255), 2 )

cv2.imshow('detected',img_bgr)

cv2.waitKey(0)
cv2.destroyAllWindows()

# # Initialize the Camera to use. 0 defaults the 1st webcam available
# cap = cv2.VideoCapture(0)
