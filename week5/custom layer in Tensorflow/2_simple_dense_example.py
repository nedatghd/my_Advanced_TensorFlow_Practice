import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

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

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype = float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)


model = tf.keras.Sequential([
                            SimpleDense(1)
                            ])

model.compile(optimizer="sgd", loss = "mean_squared_error")
model.fit(xs, ys, epochs = 500)
print(model.predict([10.0]))
print(model.variables)
