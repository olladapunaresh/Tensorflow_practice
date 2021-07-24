import tensorflow as tf
import tensorflow.keras.datasets as tfds
from tensorflow.keras.layers import Layer

class IdentityBlock(tf.keras.Model):
    def __init__(self,filters,kernel_size):
        super(IdentityBlock,self).__init__()
        self.conv1=tf.keras.layers.Conv2D(filters,kernel_size,padding='same')
        self.bn1=tf.keras.layers.BatchNormalization()

        self.conv2=tf.keras.layers.Conv2D(filters,kernel_size,padding='same')
        self.bn2=tf.keras.layers.BatchNormalization()

        self.act=tf.keras.layers.Activation('relu')
        self.add=tf.keras.layers.Add()


    def call(self, inputs, training=None, mask=None):
        x=self.conv1(inputs)
        x=self.bn1(x)
        x=self.act(x)

        x=self.conv2(x)
        x=self.bn2(x)

        x=self.add([x,inputs])

        x=self.act(x)

        return x

