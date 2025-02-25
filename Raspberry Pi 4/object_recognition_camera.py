import tensorflow as tf
import numpy as np
import cv2
import time
from PIL import Image, ImageDraw

# Preprocess images with explicit float32 type
def preprocess(image):
    image = np.array(image, dtype=np.float32)  # Explicitly set float32
    image = image - np.mean(image, dtype=np.float32)
    image /= np.std(image) + 1e-7  # Add small epsilon to avoid division by zero
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)  # Ensure output is float32

# Use OpenCV to access webcam instead of picamera
def get_camera_stream():
    cap = cv2.VideoCapture(0)  # Use default webcam
    if not cap.isOpened():
        raise Exception("Could not open webcam")
    return cap

# Create interpreter
interpreter = tf.lite.Interpreter(model_path='object_recognition.tflite')
interpreter.allocate_tensors()

# Get input and output indices
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# Print input details for debugging
input_details = interpreter.get_input_details()[0]
print(f"Input shape: {input_details['shape']}")
print(f"Input type: {input_details['dtype']}")

# Preprocess and analyze images
cap = get_camera_stream()
for i in range(10):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert from OpenCV BGR to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Crop and resize image
    h, w = frame_rgb.shape[:2]
    center_w = w // 2
    img_array = frame_rgb[:, center_w-240:center_w+240]
    img = Image.fromarray(img_array).resize((256, 256))
    
    tensor = preprocess(img)
    
    # Debug tensor type
    print(f"Tensor type: {tensor.dtype}, shape: {tensor.shape}")

    # Send image to interpreter
    interpreter.set_tensor(input_index, tensor)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)

    # Draw on image and save to file
    draw = ImageDraw.Draw(img)
    msg = 'pred: {}, prob: {:.4f}'.format(np.argmax(pred), np.max(pred))
    draw.text((10, 10), msg, fill=(255, 0, 0))
    img.save('img_{}.jpg'.format(i))
    
    # Print prediction for debugging
    print(f"Image {i} - Prediction: {np.argmax(pred)}, Confidence: {np.max(pred):.4f}")
    
    # Wait half a second
    time.sleep(0.5)

# Release the webcam
cap.release()
print("Processing complete. Check your working directory for saved images.")