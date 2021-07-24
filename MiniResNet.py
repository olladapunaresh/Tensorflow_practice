from IdentityBlock import IdentityBlock
import tensorflow as tf
import tensorflow_datasets as tfds

class MiniResNet(tf.keras.Model):
    def __init__(self,num_classes):
        super(MiniResNet, self).__init__()
        self.conv=tf.keras.layers.Conv2D(64,7,padding='same')
        self.bn=tf.keras.layers.BatchNormalization()
        self.act=tf.keras.layers.Activation('relu')
        self.max_pool=tf.keras.layers.MaxPool2D((3,3))

        self.idba1=IdentityBlock(64,3)
        self.idba2=IdentityBlock(64,3)

        self.global_pool=tf.keras.layers.GlobalMaxPool2D()
        self.classifier=tf.keras.layers.Dense(num_classes,activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x=self.conv(inputs)
        x=self.bn(x)
        x=self.act(x)
        x=self.max_pool(x)

        #insert submodels blocks as a layers

        x=self.idba1(x)
        x=self.idba2(x)

        x=self.global_pool(x)

        return self.classifier(x)


def preprocess(features):
    return tf.cast(features['image'],tf.float32)/255.0,features['label']

resnet=MiniResNet(10)
resnet.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#load and preprocess the data from mnist

dataset=tfds.load('mnist',split=tfds.Split.TRAIN)
dataset=dataset.map(preprocess).batch(32)

resnet.fit(dataset,epochs=1)

