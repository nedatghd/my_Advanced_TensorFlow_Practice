import matplotlib.pyplot as plt
import numpy as np
import os

from tensorflow.python.ops.gen_math_ops import maximum
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.python.keras.layers.core import Dropout, Lambda

from tensorflow.keras.layers import Layer
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

import tensorflow as tf

def softmax_layer(x):
    e_x = tf.exp(x)
    return e_x/tf.reduce_sum(e_x)

class SimpleDense(Layer):

    def __init__(self, units=32):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_shape[-1], self.units),
                                                  dtype="float32"), trainable=True, name="kernel")

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=w_init(shape=(1, self.units),
                                                  dtype="float32"), trainable=True, name="bias")

    def call(self, inputs):
        
        return tf.matmul(inputs, self.w) + self.b

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train, X_test = X_train/255, X_test/255

net = models.Sequential([
                        layers.Flatten(input_shape= (28 ,28)),
                        SimpleDense(units = 128),
                        layers.Lambda(lambda x:tf.maximum(x, 0.0)),
                        SimpleDense(units=10),
                        layers.Lambda(softmax_layer)
                        ])

net.compile(optimizer="sgd",
            metrics=["accuracy"],
            loss=["sparse_categorical_crossentropy"])

H = net.fit(X_train, y_train, batch_size=16, epochs=1,
            validation_data=(X_test, y_test))

plt.style.use("ggplot")
plt.plot(H.history["accuracy"], label="train")
plt.plot(H.history["val_accuracy"], label="test")
plt.plot(H.history["loss"], label="train loss")
plt.plot(H.history["val_loss"], label="test loss")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("Implent Dense layer")
plt.show()