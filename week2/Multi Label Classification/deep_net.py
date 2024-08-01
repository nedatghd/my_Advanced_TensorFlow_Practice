import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow as tf

class FashionNet():

    @staticmethod
    def build(numberCategory, numberColor):

        # input
        input_layer = layers.Input((96, 96, 3))

        # category net
        x = layers.Conv2D(32,(3,3), activation= "relu", padding="same")(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D((3, 3))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(numberCategory)(x)
        cat_net = layers.Activation("softmax", name="category_output")(x)

        # color net
        y = layers.Conv2D(16, (3, 3), activation="relu",
                          padding="same")(input_layer)
        y = layers.BatchNormalization()(y)
        y = layers.MaxPool2D((3, 3))(y)
        y = layers.Dropout(0.25)(y)

        y = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(y)
        y = layers.BatchNormalization()(y)
        y = layers.MaxPool2D((2, 2))(y)
        y = layers.Dropout(0.25)(y)

        y = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(y)
        y = layers.BatchNormalization()(y)
        y = layers.MaxPool2D((2, 2))(y)
        y = layers.Dropout(0.25)(y)

        y = layers.Flatten()(y)
        y = layers.Dense(256, activation="relu")(y)
        y = layers.Dropout(0.5)(y)
        y = layers.BatchNormalization()(y)
        y = layers.Dense(numberColor)(y)
        col_net = layers.Activation("softmax", name="color_output")(y)
        
        net = models.Model(inputs = input_layer,
                           outputs = [cat_net, col_net],
                           name = "fashionNet")
        
        return net

    @staticmethod
    def load_model():
        pass

    @staticmethod
    def save_model():
        pass
