# import cv2
# from google.colab.patches import cv2_imshow  # Import for Colab
# from ultralytics import YOLO
#
# # Load a pretrained YOLOv8n model
# model = YOLO('yolov8n.pt')
#
# # Open the video file
# video_path = "/content/IMG_8683.MP4"
# cap = cv2.VideoCapture(video_path)
#
# # Check if the video opened successfully
# if not cap.isOpened():
#     print("Error opening video file")
#     exit()
#
# # Loop through the video frames
# while(cap.isOpened()):
#     # Read a frame from the video
#     ret, frame = cap.read()
#
#     if not ret:
#         break
#
#     # Perform object detection
#     results = model(frame)
#
#     # Get the annotated frame
#     annotated_frame = results[0].plot()
#
#     # Display the resulting frame using cv2_imshow in Colab
#
#     cv2_imshow(annotated_frame)
#
#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the video capture object and destroy all windows
# cap.release()
# cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "/home/davron/PycharmProjects/python_pip_out_library/IMG_8683.MP4"  # Update the path
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Loop through the video frames
while(cap.isOpened()):
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Get the annotated frame
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()