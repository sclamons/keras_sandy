from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

os.system("taskset -p 0xff %d" % os.getpid())
matplotlib.use('Agg')

batch_size = 128
num_classes = 10
epochs = 3

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)

flat_x_train = np.array(list(map(np.ndarray.flatten, x_train)))

one_hot_labels = keras.utils.to_categorical(y_train, num_classes=num_classes)

weights = []
for i in range(2):
    model = Sequential()
    model.add(Dense(100, input_shape=(img_rows*img_cols,)))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy',
                  metrics   = ['accuracy'])
    model.fit(flat_x_train, one_hot_labels, epochs = epochs, batch_size = batch_size)
    weights.append(model.get_weights())

weights_ratio = np.abs(np.divide(weights[0], weights[1]))

def ecdf(data):
    print("data.shape: " + str(data.shape))
    x = np.sort(data)
    y = np.arange(1, 1+len(x)) / len(x)
    return x,y
flat_weights = np.concatenate(list(map(np.ndarray.flatten, weights_ratio)))
print("weights_ratio.shape = " + str(weights_ratio.shape))
x,y = ecdf(flat_weights)
plt.scatter(x,y)
plt.savefig('a_figure.png')
