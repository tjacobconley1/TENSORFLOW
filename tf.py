
import numpy
import tensorflow


# declare 'model' density and shape
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# the 'optimizer' function generates a guess
# the 'loss' function calculates the percentage difference 
# (images compared pixel by pixel??)
model.compile(optimizer='sgd', loss='mean_sqared_error')


# 6 possible values for x and y from
#        y = 2x - 1
# held in corresponding numpy arrays
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0]. dtype=float)

# check 'xs' against 'ys' 500 times using 'fit'
# this is what trains the model
model.fit(xs, ys, epochs=500)

#
# 'model.predict([a guess value])'
print(model.predict([10.0])
