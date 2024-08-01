import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from house_net import HouseNet
import pandas as pd
import matplotlib.pyplot as plt

bathroom_list = []
bedroom_list = []
frontal_list = []
kitchen_list = []
label_list = []

df = pd.read_csv("HousesInfo.txt", sep = " ",
                    names=["col_1", "col_2", "col_3", "col_4", "price"])

labels = np.array(df.loc[:, "price"])

labels = labels/np.max(labels)
print(labels)

def preprocess(img):

        img = cv2.resize(img, (32, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img/255.0

        return img

for i, item in enumerate(glob.glob("house_dataset\\*.jpg")):
        
    image = cv2.imread(item)
    image = preprocess(image)
    location = item.split("\\")[-1].split("_")[-1].split(".")[0]
    #print(location)

    if   location == "bathroom": bathroom_list.append(image)
    elif location == "bedroom": bedroom_list.append(image)
    elif location == "frontal": frontal_list.append(image)
    elif location == "kitchen": kitchen_list.append(image)

    if i % 100 == 0:
        print("[INFO]: {}/2500 processed".format(i))

bathroom_list = np.array(bathroom_list)
bedroom_list = np.array(bedroom_list)
frontal_list = np.array(frontal_list)
kitchen_list = np.array(kitchen_list)

split = train_test_split(bathroom_list, bedroom_list,
                         frontal_list, kitchen_list,labels,  test_size=0.2)

(bathroom_train, bathroom_test, bedroom_train, bedroom_test, 
 frontal_train, frontal_test, kitchen_train, kitchen_test, labels_train, labels_test) = split
print(len(labels_test))

net = HouseNet.build()

net.compile(optimizer="adam",
            loss = "MSE")

H = net.fit(x = [bathroom_train, bedroom_train, frontal_train, kitchen_train],
            y = labels_train,
            validation_data = ( [bathroom_test, bedroom_test, frontal_test, kitchen_test], labels_test),
              epochs=40,
              verbose = 1)

plt.plot(H.history["loss"], label="train loss")
plt.plot(H.history["val_loss"], label="test loss")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("accuracy/loss")
plt.title("price predictions")
plt.show()
