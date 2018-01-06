# 3. Import libraries and modules
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
import matplotlib
import matplotlib.pyplot as plt
import time
import os
from astropy.utils.data import download_file
from astropy import wcs
from astropy.io import fits
import pandas as pd 
from matplotlib.colors import LogNorm
import csv
from urllib.parse import quote
from IPython import embed
from keras.models import load_model
from datetime import datetime


np.random.seed(123)  # for reproducibility


X = np.load("data/images.npy")
y = np.load("data/ssfrs.npy")
# X = X[:100]
# y = y[:100]

train_ratio = 4/5
train_size =int(len(y)*train_ratio)

idx = np.arange(len(y), dtype = int)
np.random.shuffle(idx)

idx_train = idx[:train_size]
idx_test = idx[train_size:]

X_train = X[idx_train]
X_test = X[idx_test]
y_train = y[idx_train]
y_test = y[idx_test]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#TODO normalize data here


model = Sequential()


#AlexNet architecture: input layer, conv layer, pooling layer, conv layer, pooling layer, conv layer, conv layer, conv layer, pooling layer, fully conected layer, fully conected layer. fully conected layer, output layer 

model.add(Conv2D(32, (3, 3), activation='relu', use_bias=True, input_shape=X.shape[1:], data_format="channels_first"))
model.add(Conv2D(32, (3, 3), activation='relu', use_bias=True))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, (3, 3), activation='relu', use_bias=True))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, (3, 3), activation='relu', use_bias=True))
model.add(Conv2D(32, (3, 3), activation='relu', use_bias=True))
# model.add(Conv2D(32, (3, 3), activation='relu', use_bias=True))
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Dropout(0.25))
model.add(Flatten()) 
# Dense: a layer where each unit or neuron is connected to each neuron in the next layer   
model.add(Dense(128, activation='relu', use_bias=True))
model.add(Dense(128, activation='relu', use_bias=True))
model.add(Dense(128, activation='relu', use_bias=True))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear', use_bias=True))
 

# 8. Compile model
model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])
 

 

for i in range(50): #<-- start med 50 på cluster
    start = time.time()
    history = model.fit(X_train, y_train, validation_data = (X_test, y_test), shuffle = False,
                        batch_size=32, epochs=1, verbose=1)
    end = time.time()
    loss_history = history.history["loss"]
    numpy_loss_history = loss_history
    model.save("data/AlexNet_{}.h5".format(i))

    with open("data/AlexNet_loss_log.txt", "a") as loss_log:
        loss_log.write("{} {} {}\n".format(i, end-start, loss_history[0]))
s

predictions_train = model.predict(np.array(X_train)) 
predictions_test= model.predict(np.array(X_test))

subdir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = "data/AlexNet/predictions/{}".format(subdir)
os.makedirs("data/AlexNet/predictions", exist_ok=True)
np.save(save_path+"AlexNet_predictions_train.npy", predictions_train)
np.save(save_path+"AlexNet_predictions_test.npy", predictions_test)

np.save("data/AlexNet/X_train", X_train)
np.save("data/AlexNet/X_test", X_test)
np.save("data/AlexNet/y_train", y_train )
np.save("data/AlexNet/y_test", y_test)

#TODO: Undersøg evaluate
score = model.evaluate(X_test, y_test, verbose=0)
# score = model.predict(X_test) 
print(score)

# # 9. Fit model on training data
# #TODO: hvor starter 'fit' fra?
# model.fit(X_train, y_train, 
#           batch_size=32, epochs=10, verbose=1)

# #TODO: Undersøg evaluate
# score = model.evaluate(X_test, y_test, verbose=0)
# print(score)




