# pip install opencv-python

# import os
#
# import cv2
#
#
#
# # read image
# image_path = os.path.join('.', 'data', 'birds.jpg')
#
# img = cv2.imread(image_path)
#
# # write image
#
# cv2.imwrite(os.path.join('.', 'data', 'bird_out.jpg'), img)
#
# # visualize image
#
#
# cv2.imshow('image', img)
# cv2.waitKey(0)


import cv2

# Rasmning to'liq yo'lini ko'rsating
img = cv2.imread('/home/davron/PycharmProjects/opencv/data/bird.jpg')

# write image
files = "/home/davron/PycharmProjects/opencv/data/bird_out.jpg"
cv2.imwrite(files, img)


# visualize image
cv2.imshow("Rasm", cv2.imread(files) ) # cv2.imshow() funksiyasidan foydalaning
cv2.waitKey(0) # rasm vaqti