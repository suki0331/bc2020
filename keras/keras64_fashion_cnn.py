# 2. Complete in Sequential model
import numpy as np
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Layer, Conv2D, Dropout, Flatten, MaxPooling2D
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
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# model
model = Sequential()
model.add(Conv2D(16, kernel_size=(3,3), padding='same', input_shape=(28,28,1), activation='relu'))
model.add(Dropout(rate=0.1))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))


# compile, fit
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['acc'])
model.fit(x_train, y_train, batch_size=1024, epochs=15, verbose=2, validation_split=0.1)

# predict, evaluate
loss = model.evaluate(x_test, y_test, batch_size=32, verbose=1)
print(f"result : {loss}")

model.summary()

# result : [0.634376878029108, 0.871399998664856]







# mark acc, loss

