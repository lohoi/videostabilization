import cv2
import sys

image = cv2.imread("test.png")
cv2.imshow("Image", image)
cv2.waitKey(0)
