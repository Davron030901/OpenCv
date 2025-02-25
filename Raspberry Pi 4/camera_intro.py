# import picamera
# import picamera.array
# import cv2
# import time

# # Access camera and stream
# camera = picamera.PiCamera()
# camera.resolution = (640, 480)
# camera.brightness = 60
# camera.contrast = 20
# camera.iso = 1600
# stream = picamera.array.PiRGBArray(camera)
        
# # Capture images
# for i in range(10):
#     camera.capture(stream, format='bgr')
#     img = stream.array
#     cv2.imwrite('img_{}.jpg'.format(i), img)
#     stream.seek(0)
#     time.sleep(0.5)

# import cv2
# import time

# # Access default camera (usually webcam)
# cap = cv2.VideoCapture(0)

# # Check if camera opened successfully
# if not cap.isOpened():
#     print("Error: Could not open camera")
#     exit()

# # Set camera properties
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# try:
#     # Capture images
#     for i in range(10):
#         # Read frame
#         ret, frame = cap.read()
        
#         if ret:
#             # Save image
#             cv2.imwrite(f'img_{i}.jpg', frame)
#             print(f"Saved image {i}")
#             time.sleep(0.5)
#         else:
#             print(f"Failed to capture image {i}")

# finally:
#     # Release camera
#     cap.release()
#     print("Camera released")

import cv2
import time

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

try:
    for i in range(10):
        ret, frame = cap.read()
        
        if ret:
            # Show frame
            cv2.imshow('Camera', frame)
            
            # Save image
            cv2.imwrite(f'img_{i}.jpg', frame)
            print(f"Saved image {i}")
            
            # Wait for 500ms and check for 'q' key to quit
            if cv2.waitKey(500) & 0xFF == ord('q'):
                break
        else:
            print(f"Failed to capture image {i}")

finally:
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released")