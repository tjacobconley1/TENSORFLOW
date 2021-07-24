# PT. 3 Convolutional Neural Networks
# https://youtu.be/x_VrgWTKkiM
#         and
# https://youtu.be/u2TjZzNuly8
# Tyler Conley

# rock paper scissors - photos of hands repository
# http://bit.ly/tf-rps



import numpy as np
import tensorflow as tf
from tensorflow import keras

model = tf.keras.models.Sequential([
   # this layer specifies the input shape
   # and generate 64 filters and multiplies each of them
   # across the image
   # Then each epoch it will figure out which filters gave the best
   # signals to help match the images to their labels
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1))
   # compresses and enhances image
    tf.keras.layers.MaxPooling2D(2,2),
   #
    tf.keras.layers.Flatten(),
   #
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])



