import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Dropout,Lambda,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageFont,ImageDraw
import random

def create_pairs(x,digit_indices):
    '''
    Positive and negative pair creation.
    :param x:
    :param digit_indices:
    :return:
    '''
    pairs=[]
    labels=[]
    n=min([len(digit_indices[d]) for d in range(10)])-1

    for d in range(10):
        for i in range(n):
            z1,z2=digit_indices[d][i],digit_indices[d][i+1]
            pairs+=[[x[z1],x[z2]]]
            inc=random.randrange(1,10)
            dn=(d+inc)%10
            z1,z2=digit_indices[d][i],digit_indices[dn][i]
            pairs+=[[x[z1],x[z2]]]
            labels+=[1,0]
    return np.array(pairs),np.array(labels)

def create_pairs_on_set(images,labels):
    digit_indices=[np.where(labels==i)[0] for i in range(10)]
    pairs,y=create_pairs(images,digit_indices)
    y=y.astype('float32')
    return pairs,y


def show_image(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)
    plt.show()

#load the fashion dataset

(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()
train_images=train_images.astype('float32')
test_images=test_images.astype('float32')

#normalize the image values

train_images=train_images/255.0
test_images=test_images/255.0

#create pairs on train and test sets

tr_pairs,tr_y=create_pairs_on_set(train_images,train_labels)
ts_pairs,ts_y=create_pairs_on_set(test_images,test_labels)

#Example test
# this_pair=9
# show_image(ts_pairs[this_pair][0])
# show_image(ts_pairs[this_pair][1])
#
# print(ts_y[this_pair])

def intialize_base_network():
    input=Input(shape=(28,28,),name='base_input')
    x=Flatten(name="flatten_input")(input)
    x=Dense(128,activation='relu',name='first_base_dense')(x)
    x=Dropout(0.1,name='first_dropout')(x)
    x=Dense(128,activation='relu',name='second_base_dense')(x)
    x=Dropout(0.1,name='second_dropout')(x)
    x=Dense(128,activation='relu',name='third_base_dense')(x)

    return Model(inputs=input,outputs=x)

def eucliedean_distance(vects):
    x,y=vects
    sum_square=K.sum(K.square(x-y),axis=1,keepdims=True)
    return K.sqrt(K.maximum(sum_square,K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1,shape2=shapes
    return (shape1[0],1)

def contrastive_loss_with_margin(margin):
    def constrastive_loss(y_true,y_pred):
        square_pred=K.square(y_pred)
        margin_square=K.square(K.maximum(margin-y_pred,0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    return constrastive_loss

base_network=intialize_base_network()
# plot_model(base_network,show_shapes=True,show_layer_names=True)

#create the input and point to the base network

input_a=Input(shape=(28,28,),name='left_input')
vect_output_a=base_network(input_a)

input_b=Input(shape=(28,28,),name='right_input')
vect_output_b=base_network(input_b)
output = Lambda(eucliedean_distance,name='output_layer',output_shape=eucl_dist_output_shape)([vect_output_a,vect_output_b])

model=Model([input_a,input_b],output)

#RMSprop intialization

rms=RMSprop()

rms = RMSprop()
model.compile(loss=contrastive_loss_with_margin(margin=1), optimizer=rms)
history = model.fit([tr_pairs[:,0], tr_pairs[:,1]], tr_y, epochs=20, batch_size=128, validation_data=([ts_pairs[:,0], ts_pairs[:,1]], ts_y))