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
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
   # compresses and enhances image
    tf.keras.layers.MaxPooling2D(2,2),
   #
    tf.keras.layers.Flatten(),
   #
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# directory containing training data
TRAINING_DIR = '/tmp/faces/'

#
training_datagen = ImageDataGenerator(rescale = 1./255)

#
train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150,150),
    class_mode='categorical'
)

#
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150,150),
    class_mode='categorical'
)



# defining our neural network
model = tf.keras.models.Sequential([
   # shape is now 150x150
   # we now have 4 layers of convolution all with max pooling
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

   # Those 4 layers are then fed into the following dense layer
   # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
   # the dropout increases efficiency of the NN by eliminating some of the results
    tf.keras.layers.Dropout(0.5),

   # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')

])

   # compile the NN
    model.compile(loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])


   # fit the data
   # the training an validation datasets aren't labeled with a directory
   # because we are generating the dataset using the generators
    history = model.fit_generator(train_generator, epochs=25,
        validation_data = validation_generator,
        verbose = 1)


