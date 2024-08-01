import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import numpy as np
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import metrics

def step_one_epoch():
    step_loss = []
    for i in range(0, bat_per_epoch):
        start = i*batch_size
        end = start + batch_size
        with tf.GradientTape() as tape:
            y_prime = model(x_train[start:end])
            model_loss = categorical_crossentropy(y_train[start:end], y_prime)
    
        model_gradient = tape.gradient(model_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(model_gradient, model.trainable_variables))

        step_loss.append(model_loss)
    return np.mean(step_loss)


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train / 255).reshape((-1, 28, 28, 1))
x_test = (x_test / 255).reshape((-1, 28, 28, 1))

y_test = tf.keras.utils.to_categorical(y_test, 10)
y_train = tf.keras.utils.to_categorical(y_train, 10)

batch_size = 128
epochs = 5
bat_per_epoch = int(len(x_train) / batch_size)
bat_per_epoch_test = int(len(x_test) / batch_size)

optimizer = Adam(learning_rate=0.01, decay=0.01/epochs)

model = models.Sequential([
                          layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                          layers.Conv2D(64, (3, 3), activation='relu'),
                          layers.MaxPooling2D((2, 2)),
                          layers.Dropout(0.25),
                          layers.Flatten(),
                          layers.Dense(128, activation='relu'),
                          layers.Dense(32, activation='relu'),
                          layers.Dropout(0.5),
                          layers.Dense(10, activation="softmax")
                          ])

train_loss = []

for epoch in range(epochs):
    epochStart = time.time()

    out_loss = step_one_epoch()
    train_loss.append(out_loss)

    epochEnd = time.time()
    elapsed = (epochEnd - epochStart)
    print(f"epoch: {epoch+1}, took: {elapsed:.2f}, loss: {out_loss:.2f}")

model.compile(optimizer=optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])
print("accuracy:", model.evaluate(x_test, y_test, verbose = 0)[1])

plt.plot(range(epochs), train_loss, label = "train loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()