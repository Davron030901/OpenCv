# from PIL import Image
# import tflite_runtime.interpreter as tflite
# import numpy as np
# from os import listdir

# # Preprocess images
# def preprocess(img_dir, img_file):
#     img = np.array(Image.open(img_dir + '/' + img_file))
#     img = img.astype(np.float32) - np.mean(img)
#     img /= np.std(img)
#     img = np.expand_dims(img, axis=0)
#     return img

# # Create interpreter
# ip = tflite.Interpreter(model_path='object_recognition.tflite')
# ip.allocate_tensors()

# # Get input/output indices
# input_index = ip.get_input_details()[0]['index']
# output_index = ip.get_output_details()[0]['index']

# # Load test images
# img_dir = 'test_imgs'
# num_correct = 0
# for img_file in listdir(img_dir):
    
#     # Send image to interpreter
#     img = preprocess(img_dir, img_file)
#     ip.set_tensor(input_index, img)

#     # Launch interpreter and get prediction
#     ip.invoke()
#     preds = ip.get_tensor(output_index)
    
#     # Test classifications
#     label = int(img_file.split('_')[0])
#     if np.argmax(preds) == label:
#         num_correct += 1

# # Display accuracy
# num_imgs = len(listdir(img_dir))
# print('{} correct out of {}'.format(num_correct, num_imgs))

from PIL import Image
import tensorflow as tf
import numpy as np
from os import listdir

def preprocess(img_dir, img_file):
    try:
        # Read image
        img = Image.open(img_dir + '/' + img_file)
        
        # Convert to numpy array with explicit float32
        img = np.array(img, dtype=np.float32)
        
        # Print shape and dtype for debugging
        print(f"Image shape: {img.shape}, dtype: {img.dtype}")
        
        # Normalize
        mean = np.mean(img, dtype=np.float32)
        std = np.std(img, dtype=np.float32)
        img = (img - mean) / std
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Force float32 type
        return tf.cast(img, tf.float32).numpy()
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

# Create interpreter and get details
ip = tf.lite.Interpreter(model_path='object_recognition.tflite')
ip.allocate_tensors()

# Print model input details
input_details = ip.get_input_details()
output_details = ip.get_output_details()

print("\nModel Input Details:")
print(f"Shape: {input_details[0]['shape']}")
print(f"Type: {input_details[0]['dtype']}")

input_index = input_details[0]['index']
output_index = output_details[0]['index']

# Process images
img_dir = 'test_imgs'
num_correct = 0

try:
    for img_file in listdir(img_dir):
        print(f"\nProcessing {img_file}")
        
        # Preprocess image
        img = preprocess(img_dir, img_file)
        if img is None:
            continue
            
        print(f"Processed image shape: {img.shape}, dtype: {img.dtype}")
        
        # Verify tensor type
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        
        # Set tensor
        ip.set_tensor(input_index, img)
        
        # Run inference
        ip.invoke()
        
        # Get prediction
        preds = ip.get_tensor(output_index)
        
        # Test classifications
        label = int(img_file.split('_')[0])
        prediction = np.argmax(preds)
        
        print(f"Predicted: {prediction}, Actual: {label}")
        
        if prediction == label:
            num_correct += 1

    # Display accuracy
    num_imgs = len(listdir(img_dir))
    print('\nResults:')
    print(f'{num_correct} correct out of {num_imgs}')
    print(f'Accuracy: {(num_correct/num_imgs * 100):.2f}%')

except Exception as e:
    print(f"\nError during execution: {e}")