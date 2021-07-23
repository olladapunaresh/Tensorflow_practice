from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense , Flatten
from tensorflow.keras.models import Model
input = Input(shape=(28,28))

x=Flatten()(input)
x= Dense(128,activation="relu")(x)
predictions = Dense(10,activation="softmax")(x)


func_model=Model(inputs=input,outputs=predictions)
