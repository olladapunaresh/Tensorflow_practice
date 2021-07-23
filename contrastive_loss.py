from tensorflow.keras.losses import Loss
import tensorflow as tf
from tensorflow.keras import backend as K
class ContrativeLoss(Loss):
    def __init__(self,margin):
        super().__init__()
        self.margin=margin

    def call(self, y_true, y_pred):
        square_pred=K.square(y_pred)
        margin_square=K.square(K.maximum(self.margin-y_pred,0))
        return K.mean(y_true*square_pred+(1-y_true)*margin_square)
