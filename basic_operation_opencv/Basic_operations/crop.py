# crop qirqish
import os

import cv2


img = cv2.imread(os.path.join('.', 'dog.jpg'))

print(img.shape)

cropped_img = img[20:540, 480:1100]# h--w

cv2.imshow('img', img)
cv2.imshow('cropped_img', cropped_img)
cv2.waitKey(0)