import cv2
import matplotlib.pyplot as plt

cb_img=cv2.imread("/home/davron/PycharmProjects/opencv_lesson/checkerboard_color.png")
coke_img=cv2.imread("/home/davron/PycharmProjects/opencv_lesson/coca-cola-logo.png")

# Use motplotlib imshow()
plt.imshow(cb_img)
plt.title("matplotlib imshow")
plt.show()

# Use OpenCV  imshow() , dicplay for 8 sec
window1=cv2.namedWindow("w1")
cv2.imshow(window1,cb_img)
cv2.waitKey(8000)
cv2.destroyWindow(window1)

# Use OpenCV  imshow() , dicplay for 8 sec
window2=cv2.namedWindow("w2")
cv2.imshow(window2,coke_img)
cv2.waitKey(8000)
cv2.destroyWindow(window2)

# Use OpenCV  imshow() , dicplay for 8 sec
window3=cv2.namedWindow("w3")
cv2.imshow(window3,cb_img)
cv2.waitKey(0)
cv2.destroyWindow(window3)

window4=cv2.namedWindow("w4")

Alive=True

while Alive:
    # Use OpenCV  imshow() ,display until 'q' key is pressed
    cv2.imshow(window4, coke_img)
    keyword=cv2.waitKey(1)

    if keyword==ord("q"):
        Alive=False
    cv2.destroyWindow(window4)
cv2.destroyAllWindow()
stop=1

