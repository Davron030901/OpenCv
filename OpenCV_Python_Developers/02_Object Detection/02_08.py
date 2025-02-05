import numpy as np
import cv2

img = cv2.imread("fuzzy.png",1)
cv2.imshow("Original", img)


cv2.waitKey(0)
cv2.destroyAllWindows()