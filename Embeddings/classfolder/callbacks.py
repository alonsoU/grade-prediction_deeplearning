import tensorflow as tf
import numpy as np
import scipy as sp
from tensorflow import keras

class MyCustomCallback(tf.keras.callbacks.Callback):

  def on_epoch_begin(self, epoch, logs=None):
    self.model.layers[2].stddev = random.uniform(0, 1)
    print('updating sttdev in training')
    print(self.model.layers[2].stddev)

class ThresholdAcc(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    target_accuracy = 0.99
    if logs.get('acc')>target_accuracy:
      print(f"\nReached {target_accuracy*100}% accuracy so cancelling training!")
      self.model.stop_training = True

class
