import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, Input

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
# model = Sequential()
# # model.add(Dropout(0.4, input_shape=(28,28,1)))
# model.add(Conv2D(16, (5,5), padding='same', input_shape=(28,28,1), activation='relu')) 
# model.add((Dropout(0.4)))
# model.add(Conv2D(32, (3,3), padding='same', activation='relu')) 
# model.add(MaxPooling2D(pool_size=3))                    
# model.add((Dropout(0.4)))
# model.add(Conv2D(64, (2,2), padding='same', activation='relu')) 
# model.add((Dropout(0.4)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(10, activation='softmax'))

input1 = Input(shape=(28,28,1))
layer1 = Conv2D(filters=16, kernel_size=(5,5), padding='same', activation='relu')(input1)
layer2 = Dropout(rate=0.4)(layer1)
layer3 = Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(layer2)
layer4 = MaxPooling2D(pool_size=(3,3))(layer3)
layer5 = Dropout(rate=0.4)(layer4)
layer6 = Conv2D(filters=64, kernel_size=(2,2), padding='same', activation='relu')(layer5)
layer7 = Dropout(rate=0.4)(layer6)
layer8 = Flatten()(layer7)
dense1 = Dense(128)(layer8)
output1 = Dense(10, activation='softmax')(dense1)

model = Model(inputs=input1, output=output1)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, batch_size=52, epochs =20, verbose=1)

# evaluate, predict
loss, acc = model.evaluate(x_test, y_test, batch_size=512)
print(f"loss : {loss} \n acc : {acc}")
model.summary()


