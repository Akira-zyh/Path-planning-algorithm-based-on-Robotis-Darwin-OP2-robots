import cv2
import os
import numpy as np
folder_path = './data/easy1/'

img1 = cv2.imread(folder_path + 'easy1_104.png')
hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
lower_green = np.array([35, 128, 46])  # 提高饱和度下限减少干扰
upper_green = np.array([77, 255, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)

cv2.imshow("Image Window",mask)
cv2.waitKey(0)