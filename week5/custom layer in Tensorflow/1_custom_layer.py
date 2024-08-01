import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.layers import Layer
import tensorflow as tf

class SimpleDense(Layer):

    def __init__(self, units = 32):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_shape[-1], self.units),
                                                  dtype="float32"), trainable=True, name="kernel")
        
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(1, self.units),
                                dtype="float32"), trainable=True, name="bias")

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

my_dense = SimpleDense(units=1)
x = tf.ones((1,1))
y = my_dense(x)
print(my_dense.variables)
print(y)