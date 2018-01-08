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
 

# 6. Preprocess class labels
#Y_train = np_utils.to_categorical(y_train, 10)
#Y_test = np_utils.to_categorical(y_test, 10)
 
# 7. Define model architecture (We have chosen Sequential instead of Model)
model = Sequential()

#embed()
#LeNet architecture: input layer, conv layer, pooling layer, conv layer, pooling layer, fully conected layer, output layer 

#input layer
model.add(Conv2D(32, (3, 3), activation='relu', use_bias=True, input_shape=X.shape[1:], data_format="channels_first"))
# 1st convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu', use_bias=True))
# 1st pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten()) 
# fully connected layer
model.add(Dense(128, activation='relu', use_bias=True))
model.add(Dropout(0.5))
# output layer 
model.add(Dense(1, activation='linear', use_bias=True))
 
# 8. Compile model
model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])
 
# 9. Fit model on training data
#TODO: hvor starter 'fit' fra?

for i in range(50): #<-- start med 50 på cluster
    start = time.time()
    history = model.fit(X_train, y_train, validation_data = (X_test, y_test), shuffle = False,
                        batch_size=32, epochs=1, verbose=1)
    end = time.time()
    loss_history = history.history["loss"]
    numpy_loss_history = loss_history
    model.save("data/LeNet_{}.h5".format(i))

    with open("data/LeNet_loss_log.txt", "a") as loss_log:
        loss_log.write("{} {} {}\n".format(i, end-start, loss_history[0]))
    

predictions_train = model.predict(np.array(X_train)) 
predictions_test= model.predict(np.array(X_test))

subdir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = "data/LeNet/predictions/{}".format(subdir)
os.makedirs("data/LeNet/predictions", exist_ok=True)
np.save(save_path+"LeNet_predictions_train.npy", predictions_train)
np.save(save_path+"LeNet_predictions_test.npy", predictions_test)

np.save("data/LeNet/X_train", X_train)
np.save("data/LeNet/X_test", X_test)
np.save("data/LeNet/y_train", y_train )
np.save("data/LeNet/y_test", y_test)


#TODO: Undersøg evaluate
score = model.evaluate(X_test, y_test, verbose=0)
print(score)

