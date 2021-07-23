import tensorflow as tf
import numpy as np
from tensorflow import keras
from my_huber_loss import MyHuberLoss
# inputs
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)

# labels
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)


print(xs)

model=tf.keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
model.compile(optimizer='sgd',loss='mean_squared_error')
model.fit(xs,ys,epochs=500,verbose=0)
print(model.predict([10.0]))


def custom_huber_loss(y_true,y_pred):
    threshold=1
    error= y_true-y_pred
    is_small_error = tf.abs(error)<= threshold
    small_error_loss=tf.square(error)/2
    big_error_loss=threshold*(tf.abs(error)-0.5*threshold)
    return tf.where(is_small_error,small_error_loss,big_error_loss)

def my_huber_loss_with_threshold(threshold):
    def custom_huber_loss(y_true,y_pred):
        error=y_true-y_pred
        is_smalll_error=tf.abs(error)<=threshold
        small_error_loss=tf.square(error)/2
        big_error_loss=threshold*(tf.abs(error)-0.5*threshold)
        return tf.where(is_smalll_error,small_error_loss,big_error_loss)
    return custom_huber_loss

model=tf.keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
model.compile(optimizer='sgd',loss=MyHuberLoss(threshold=1.2))
model.fit(xs,ys,epochs=500,verbose=0)

print(model.predict([10.0]))