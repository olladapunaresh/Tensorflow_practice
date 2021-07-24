import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

class SimpleDense(Layer):

    def __init__(self,units,activation=None):
        super(SimpleDense,self).__init__()
        self.units=units
        self.activation=tf.keras.activations.get(activation)

    def build(self,input_shape):
        w_init=tf.random_normal_initializer()
        #initialize the weights
        self.w=tf.Variable(name='kernel',
                           initial_value=w_init(shape=(input_shape[-1],self.units),dtype='float32'),trainable=True)

        #initialize the biases

        b_init=tf.zeros_initializer()
        self.b=tf.Variable(name='bias',
                           initial_value=b_init(shape=(self.units,),dtype='float32'),trainable=True)

    def call(self, inputs):
        return self.activation(tf.matmul(inputs,self.w)+self.b)







