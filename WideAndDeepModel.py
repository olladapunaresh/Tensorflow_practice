from tensorflow.keras import Model
from tensorflow.keras.layers import Dense , concatenate


class WideAndDeepModel(Model):
    def __init__(self,units=30,activation=None):
        super(WideAndDeepModel, self).__init__()
        self.hidden_1=Dense(units,activation=activation)
        self.hidden_2=Dense(units,activation=activation)
        self.main_output=Dense(1)
        self.aux_output=Dense(1)

    def call(self, inputs, training=None, mask=None):
        input_a,input_b=inputs
        hidden1=self.hidden_1(input_b)
        hidden2=self.hidden_2(hidden1)
        concat=concatenate([input_a,hidden2])
        main_output=self.main_output(concat)
        aux_output=self.aux_output(hidden2)

        return main_output,aux_output

model=WideAndDeepModel()
