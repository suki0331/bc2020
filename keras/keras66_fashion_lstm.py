# 2. Complete in Sequential model
import numpy as np
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Layer, LSTM, Dropout
from keras.utils import np_utils

# get data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# always check data shape!!
print(x_train.shape) # (60000, 28, 28)
print(x_test.shape)  # (10000, 28, 28)
print(y_train.shape) # (60000,)
print(y_test.shape)  # (10000,)

print(x_train[0])
print(y_train[0])

# data preprocessing 1. one-hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


print(y_train)
print(y_test)

# data preprocessing 2. normalization
x_train = x_train.reshape(60000, 1, 784)/255.
x_test = x_test.reshape(10000, 1, 784)/255.

# model
model = Sequential()
model.add(LSTM(32, input_shape=(1, 784), activation='relu'))
model.add(Dropout(rate=0.1))
model.add(Dense(10, activation='softmax'))


# compile, fit
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['acc'])
model.fit(x_train, y_train, batch_size=1024, epochs=15, verbose=2, validation_split=0.1)

# predict, evaluate
loss = model.evaluate(x_test, y_test, batch_size=32, verbose=1)
print(f"result : {loss}")

model.summary()

# mark acc, loss
# result : [0.3920144690990448, 0.8610000014305115]