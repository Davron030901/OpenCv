# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers, models
# from keras.preprocessing import image # type: ignore
# from os import listdir

# # Preprocess images
# def preprocess(img_dir, img_file):
#     img = image.img_to_array(image.load_img(img_dir + '/' + img_file))    
#     img = img.astype(np.float64) - np.mean(img)
#     img /= np.std(img)
#     return img

# # Load training data
# def load_data(img_dir, num_classes):
#     for img_file in listdir(img_dir):
#         imgs = []
#         labels = []
#         img = preprocess(img_dir, img_file)
#         imgs.append(img)
#         label = int(img_file.split('_')[0])
#         labels.append(np.eye(num_classes)[label])
#         return imgs, labels

# # Initialize the model
# num_classes = 4
# imgs, labels = load_data('train_imgs', num_classes)
# model = models.Sequential()
# model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=imgs[0].shape))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(16, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(num_classes, activation='softmax'))

# # Compile the model
# model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(np.asarray(imgs), np.asarray(labels), epochs=10, verbose=0)

# # Save the model in TensorFlow Lite format
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# open('object_recognition.tflite', 'wb').write(converter.convert())

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import listdir
import PIL

# Preprocess images
def preprocess(img_dir, img_file):
    img_path = os.path.join(img_dir, img_file)
    img = image.load_img(img_path)
    img = image.img_to_array(img)
    img = img.astype(np.float64) - np.mean(img)
    img /= np.std(img)
    return img

# Load training data
def load_data(img_dir, num_classes):
    imgs = []
    labels = []
    
    if not os.path.exists(img_dir):
        raise Exception(f"Directory '{img_dir}' does not exist")
        
    for img_file in listdir(img_dir):
        try:
            img = preprocess(img_dir, img_file)
            imgs.append(img)
            label = int(img_file.split('_')[0])
            labels.append(np.eye(num_classes)[label])
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            
    if not imgs:
        raise Exception("No images were loaded")
        
    return np.array(imgs), np.array(labels)

try:
    # Initialize the model
    num_classes = 4
    
    print("Loading training data...")
    imgs, labels = load_data('train_imgs', num_classes)
    print(f"Loaded {len(imgs)} images")

    print("Creating model...")
    model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=imgs[0].shape),
    layers.Dropout(0.25),  # Dropout qo'shish
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.Dropout(0.25),  # Dropout qo'shish
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(num_classes, activation='softmax')
])
    datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
    # Compile the model
    print("Compiling model...")
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

    # Train the model
    print("Training model...")
    history = model.fit(
        imgs, 
        labels,
        epochs=10,
        validation_split=0.2,
        verbose=1
    )

    # Save the model in TensorFlow Lite format
    print("Converting model to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open('object_recognition.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Model saved as object_recognition.tflite")

except Exception as e:
    print(f"An error occurred: {e}")