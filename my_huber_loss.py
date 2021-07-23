from tensorflow.keras.losses import Loss
import tensorflow as tf
class MyHuberLoss(Loss):
    threshold=1

    def __init__(self,threshold):
        super().__init__()
        self.threshold=threshold

    def call(self, y_true, y_pred):
        error=y_true-y_pred
        is_small_error=tf.abs(error)<=self.threshold
        small_error_loss=tf.square(error)/2
        big_error_loss=self.threshold*(tf.abs(error)-0.5*self.threshold)
        return tf.where(is_small_error,small_error_loss,big_error_loss)

