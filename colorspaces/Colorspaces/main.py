import os

import cv2
#openCV color spaces

img = cv2.resize(cv2.imread(os.path.join('.', 'bird.jpg')),(500,500))

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


cv2.imshow('img', img)
cv2.imshow('img_gray', img_gray)
cv2.imshow('img_rgb', img_rgb)
cv2.imshow('img_hsv', img_hsv)
cv2.waitKey(0)#paydo bo'lish vaqri