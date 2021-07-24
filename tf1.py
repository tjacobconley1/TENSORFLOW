
import tensorflow as tf
from tensorflow import keras


# the fashion_mnist dataset is build into tensorflow
#
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# takes in a 28x28 set of pixels and outputs 1 of 10 values
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
   # activation functions
    keras.layers.Dense(128, activation=tf.nn.relu),  # 'rectified linear unit (returns a value only if it's larger than 0
   # picking the largest number in the set
   # returns a value 1 - 10
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# the 'optimizer' function generates a guess
# the 'loss' function calculates the percentage difference
# (images compared pixel by pixel??)
model.compile(optimizer='sgd', loss='mean_squared_error')

# fit the training images to the training labels
model.fit(train_images, train_labels, epochs=5)

# test results against datasets that it hasn't seen
# the 'model.evaluate()' function
#test_loss, test_acc = model.evaluate(test_images, test_labels)

# predictions for possible new images
# the 'model.predict()' function
#predictions = model.predict(my_images)


#print(predictions)
