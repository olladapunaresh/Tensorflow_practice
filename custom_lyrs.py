import tensorflow as tf
from SimpleDense import SimpleDense
import numpy as np

my_dense=SimpleDense(units=1,activation='relu')

x=tf.ones((1,1))
y=my_dense(x)


xs = np.array([1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([1.0, 0.0, 1.0, 4.0, 6.0, 8.0], dtype=float)

model=tf.keras.Sequential([my_dense])

model.compile(loss='mean_squared_error',optimizer='sgd')
model.fit(xs,ys,epochs=500,verbose=0)

print(model.predict([10.0]))

print(my_dense.variables)

