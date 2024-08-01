import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras import layers, models

class HouseNet():

    @staticmethod
    def build():
        bathroom_input = layers.Input((32, 32, 3))

        x = layers.Conv2D(16, (3, 3), padding="same",activation="relu")(bathroom_input)
        x = layers.MaxPool2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
        x = layers.MaxPool2D((2, 2))(x)

        bedroom_input = layers.Input((32, 32, 3))
        y = layers.Conv2D(16, (3, 3), padding="same",
                          activation="relu")(bedroom_input)
        y = layers.MaxPool2D((2, 2))(y)
        y = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(y)
        y = layers.MaxPool2D((2, 2))(y)

        frontal_input = layers.Input((32, 32, 3))
        z = layers.Conv2D(16, (3, 3), padding="same",
                          activation="relu")(frontal_input)
        z = layers.MaxPool2D((2, 2))(z)
        z = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(z)
        z = layers.MaxPool2D((2, 2))(z)

        kitchen_input = layers.Input((32, 32, 3))
        w = layers.Conv2D(16, (3, 3), padding="same",
                          activation="relu")(kitchen_input)
        w = layers.MaxPool2D((2, 2))(w)
        w = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(w)
        w = layers.MaxPool2D((2, 2))(w)
        

        concat_inputs = layers.concatenate([x, y, z, w], axis = 2)
        flat_layer = layers.Flatten()(concat_inputs)
        out = layers.Dense(100, activation="relu")(flat_layer)
        out = layers.Dense(1, activation="linear")(out)

        net = models.Model(
            inputs=[bathroom_input, bedroom_input, frontal_input, kitchen_input], outputs=out)

        return net

    @staticmethod
    def load_model():
        pass

    @staticmethod
    def save_model():
        pass