import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# always check shape!!
print(x_train.shape)
print(y_train.shape)

# y_train = y_train.reshape(60000,1) # 딱히 의미없음
# y_test = y_test.reshape(10000,1) # 딱히 의미없음

# 1. one_hot_encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape) # 60000, 10

# 2. normalization
x_train = x_train.reshape(60000,2,392)/255.
x_test = x_test.reshape(10000,2,392)/255.
# x_train = x_train.reshape(60000,28,28)/255.
# x_test = x_test.reshape(10000,28,28)/255.

# model
model = Sequential()
model.add(LSTM(32, activation='relu', input_shape=(2,392)))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, batch_size=1000, epochs=15)

# evaluate, predict
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print(f"loss : {loss} \n acc : {acc}")

model.summary()
