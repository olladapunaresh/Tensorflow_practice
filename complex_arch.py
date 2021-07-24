import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import plot_model

#inputs to the model

input_a=Input(shape=[1],name='Wide Input')
input_b=Input(shape=[1],name='Deep Input')

#define deep depth

hidden_1=Dense(30,activation='relu')(input_b)
hidden_2=Dense(30,activation='relu')(hidden_1)

#define merged path

concat=concatenate([input_a,hidden_2])
output=Dense(1,name='output')(concat)

#define the second output for the model

aux_output=Dense(1,name='aux_output')(hidden_2)

model=Model(inputs=[input_a,input_b],outputs=[output,aux_output])
#currently not working in the pycharm but please enable it when writing the code in jupyter notebook
#plot_model(model)


