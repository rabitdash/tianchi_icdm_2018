import tensorflow as tf
from tensorflow.contrib import keras


class Model(keras.layers.GRU):
    def __init__(self):
        super.__init__()
