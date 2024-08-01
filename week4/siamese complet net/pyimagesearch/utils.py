# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# -----------------------------
#   FUNCTIONS
# -----------------------------
def make_pairs(images, labels):
    # Initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive or negative
    pairImages = []
    pairLabels = []
    # Calculate the total number of classes present in the dataset and then build a list of indexes
    # for each class label that provides the indexes for all examples with a given label
    numClasses = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range(0, numClasses)]
    # Loop over all images
    for idxA in range(len(images)):
        # Grab the current image and label belonging to the current iteration
        currentImage = images[idxA]
        label = labels[idxA]
        # Randomly pick an image that belongs to the *same* class label
        idxB = np.random.choice(idx[label])
        posImage = images[idxB]
        # Prepare a positive pair and update the images and labels lists, respectively
        pairImages.append([currentImage, posImage])
        pairLabels.append([1])
        # Grab the indices for each of the class labels *not* equal to the current label and randomly pick an image
        # corresponding to a label *not* equal to the current label
        negIdx = np.where(labels != label)[0]
        negImage = images[np.random.choice(negIdx)]
        # Prepare a negative pair of images and update our lists
        pairImages.append([currentImage, negImage])
        pairLabels.append([0])
    # Return a 2-tuple of the image pairs and labels
    return np.array(pairImages), np.array(pairLabels)


def euclidean_distance(vectors):

    (featsA, featsB) = vectors

    return tf.math.reduce_euclidean_norm(featsA - featsB, axis = 1, keepdims = True)

def plot_training(H, plotPath):
    # Construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.plot(H.history["accuracy"], label="train_acc")
    plt.plot(H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)


def contrastive_loss(y, preds):
    margin = 1
    # explicitly cast the true class label data type to the predicted
    # class label data type (otherwise we run the risk of having two
    # separate data types, causing TensorFlow to error out)
    y = tf.cast(y, preds.dtype)
    # calculate the contrastive loss between the true labels and
    # the predicted labels
    squaredPreds = K.square(preds)
    squaredMargin = K.square(K.maximum(margin - preds, 0))
    loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
    # return the computed contrastive loss to the calling function
    return loss
