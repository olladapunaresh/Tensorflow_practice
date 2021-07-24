import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten,Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from utils import *
#use pandas excel reader
df=pd.read_csv('ENB2012_data.csv')
del df['Unnamed: 10']
del df['Unnamed: 11']

df=df.sample(frac=1).reset_index(drop=True)
df.dropna(inplace=True)
train,test=train_test_split(df,test_size=0.2)
train_stats=train.describe()
#get Y1 and y2 as 2 outputs and format them as np arrays

train_stats.pop('Y1')
train_stats.pop('Y2')
train_stats=train_stats.transpose()
train_Y=format_output(train)

test_Y=format_output(test)

norm_train_X=norm(train,train_stats)
norm_test_X=norm(test,train_stats)

#build the model using fucntional syntax in tensorflow

input_layer=Input(shape=(len(train.columns),))
first_dense=Dense(units='128',activation='relu')(input_layer)
second_dense=Dense(units='128',activation='relu')(first_dense)

#Y1 output will be directly fed from the second dense
y1_output=Dense(units='1',name='y1_output')(second_dense)
third_dense=Dense(units='64',activation='relu')(second_dense)

#Y2 output will be connected from the third dense layers

y2_output=Dense(units='1',name='y2_output')(third_dense)

model=Model(inputs=input_layer,outputs=[y1_output,y2_output])

print(model.summary())


#Configure the parameters

optimizer=tf.keras.optimizers.SGD(lr=0.001)

model.compile(optimizer=optimizer,loss={'y1_output':'mse','y2_output':'mse'},
              metrics={
                  'y1_output':tf.keras.metrics.RootMeanSquaredError(),
                  'y2_output':tf.keras.metrics.RootMeanSquaredError()
              })
tf.keras.metrics.BinaryCrossentropy

history=model.fit(norm_train_X,train_Y,epochs=500,batch_size=10,validation_data=(norm_test_X,test_Y))

loss,Y1_loss,Y2_loss,Y1_rmse,Y2_rmse= model.evaluate(x=norm_test_X,y=test_Y)
print("Loss = {}, Y1_loss = {}, Y1_mse = {}, Y2_loss = {}, Y2_mse = {}".format(loss, Y1_loss, Y1_rmse, Y2_loss, Y2_rmse))

Y_pred=model.predict(norm_test_X)
plot_diff(test_Y[0],Y_pred[0],title='Y1')
plot_diff(test_Y[1],Y_pred[1],title='Y2')
plot_metrics(metric_name='y1_output_root_mean_squared_error',title='Y1 RMSE',ylim=6,history=history)
plot_metrics(metric_name='y1_output_root_mean_squared_error',title='Y2 RMSE',ylim=7,history=history)
