import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from deep_net import FashionNet
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
plt.style.use("ggplot")

all_images = []
category_labels = []
color_labels = []

for i , item in enumerate(glob.glob("dataset\\*\\*")):

    img = cv2.imread(item)
    img = cv2.resize(img, (96, 96))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    all_images.append(img)

    (color, category) = item.split("\\")[-2].split("_")
    category_labels.append(category)
    color_labels.append(color)

    if i % 100 == 0:
        print("[INFO]: {}/2500 processed".format(i))

all_images = np.array(all_images, dtype="float")/255.0

color_labels = np.array(color_labels)
category_labels = np.array(category_labels)

categoryLB = LabelBinarizer()
colorLB = LabelBinarizer()
category_labels = categoryLB.fit_transform(category_labels)
color_labels = colorLB.fit_transform(color_labels)

split = train_test_split(all_images, category_labels,
                         color_labels,  test_size=0.2)

(trainX, testX, trainCategoryY, testCategoryY, trainColorY, testColorY) = split

fnet = FashionNet()

net = fnet.build(len(categoryLB.classes_), len(colorLB.classes_))

losses = {
        "category_output": "categorical_crossentropy",
        "color_output": "categorical_crossentropy",
        }

loss_weights = {"category_output": 1.0, "color_output": 1.0}

net.compile(optimizer="adam",
            loss = losses, 
            loss_weights = loss_weights,
            metrics = ["accuracy"])

H = net.fit(x=trainX,
            y = {"category_output": trainCategoryY, "color_output": trainColorY},
            validation_data=(testX,
                             {"category_output": testCategoryY, "color_output": testColorY}),
              epochs=10,
              verbose = 1)

plt.plot(H.history["category_output_accuracy"], label = "category acc")
plt.plot(H.history["val_category_output_accuracy"], label="val category acc")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()
plt.close()