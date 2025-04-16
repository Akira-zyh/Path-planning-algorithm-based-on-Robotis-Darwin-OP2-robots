import cv2
import os
import numpy as np
folder_path = './data/easy1/'

img = cv2.imread(folder_path + 'easy1_104.png')
height, width = img.shape[:2]
mask = np.zeros((height, width), dtype=np.uint8)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_green = np.array([35, 128, 46])  # 提高饱和度下限减少干扰
upper_green = np.array([77, 255, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)
blurred = cv2.GaussianBlur(mask, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
blank_image = np.zeros_like(img)
cv2.drawContours(blank_image, contours, -1, (0, 0, 255), 2)


cv2.imshow("Image Window", blank_image)
cv2.waitKey(0)
