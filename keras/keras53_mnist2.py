import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

# load proprcessed data from mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train[0])  # image. 0~255 
# print(f"y_train[0] : {y_train[0]}")

print(x_train.shape)     # (60000, 28, 28) 60000 28*28 images
print(x_test.shape)      # (10000, 28, 28) 10000 28*28 images  
print(y_train.shape)     # (60000,)  60000 scalars
print(y_test.shape)      # (10000,)  10000 scalars

# plt.imshow(x_train[33123], 'inferno_r')  # imshow == show images
# plt.imshow(x_train[0])  
# plt.show()

# data preprocessing 1. one_hot_encoding
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)  # always check the shape!!

# data preprocessing 2. normalization
x_train = x_train.reshape(60000, 28, 28, 1)/255.
x_test = x_test.reshape(10000, 28, 28, 1)/255.
# x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255 # or /255. , /255.0
# x_test = x_test.reshape(60000, 28, 28, 1).astype('float32')/255

# compile, fit
model = Sequential()
model.add(Conv2D(32, (5,5), input_shape=(28,28,1), activation='relu')) 
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, batch_size=1024, epochs =10, verbose=1)


# evaluate, predict
loss, acc = model.evaluate(x_test, y_test, batch_size=1024)
print(f"loss : {loss} \n acc : {acc}")
model.summary()


