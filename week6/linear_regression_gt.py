import os
from matplotlib import markers
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import matplotlib.pyplot as plt

w = tf.Variable(2.0)
b = tf.Variable(1.0)
TRUE_w = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 100

#inputs : xs
#output : real output
#predicted_outputs: predicted output

def plot_data(inputs, outputs, predicted_outputs):
    plt.scatter(inputs, outputs, c = "b", marker=".", label = "Real Data")
    plt.scatter(inputs, predicted_outputs, c="r", marker="+", label="Predicted Data")
    plt.legend()
    plt.show()

def loss(y_pred, y_true):
    loss = tf.reduce_mean(tf.square(y_pred - y_true))
    return loss

def train(outputs, learning_rate):
   
    with tf.GradientTape() as tape:
        y_pred = w*xs + b
        current_loss = loss(y_pred, outputs)
    
    dw, db = tape.gradient(current_loss, [w, b])
    w.assign_sub(learning_rate*dw)
    b.assign_sub(learning_rate*db)

    return current_loss


def plot_loss_for_weights(weights_list, losses):
    for idx, weights in enumerate(weights_list):
        plt.subplot(120+idx+1)
        plt.plot(weights["values"], losses, "r")
        plt.plot(weights["values"], losses, "bo")
        plt.xlabel(weights["name"])
        plt.ylabel("loss")
    
    plt.show()

xs = tf.random.normal(shape = (NUM_EXAMPLES,))
ys = TRUE_w*xs + TRUE_b

predicted_output = w*xs + b
plot_data(xs, ys, predicted_output)

list_w, list_b = [], []
epochs = range(100)
losses = []

for epoch in epochs:
    list_w.append(w.numpy())
    list_b.append(b.numpy())
    current_loss = train(ys, 0.01)
    losses.append(current_loss)
    print(f"epoch: {epoch}, w= {list_w[-1]:.2f}, b={list_b[-1]:.2f}, loss= {current_loss:.5f}")

plt.plot(epochs, list_w, c= "r")
plt.plot(epochs, list_b, c="b")
plt.plot(epochs, [TRUE_w]*len(epochs), c="r", marker = ".")
plt.plot(epochs, [TRUE_b]*len(epochs), c="b", marker=".")

plt.show()

test_inputs = tf.random.normal(shape = (NUM_EXAMPLES,))
test_outputs = test_inputs*TRUE_w + TRUE_b

predicted_outputs = test_inputs*w + b
plot_data(test_inputs, test_outputs, predicted_outputs)

weights_list = [{"name": "w", "values": list_w},
                {"name": "b", "values": list_b}]


plot_loss_for_weights(weights_list, losses)

