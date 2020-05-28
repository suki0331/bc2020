# download dataset from keras.datasets
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Flatten
from keras.layers import Dropout, MaxPooling2D, Input
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Data preprocessing 1. one-hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)

# Data preprocessing 2. normalization
x_train = x_train.reshape(50000,3,1024)/255.0
x_test = x_test.reshape(10000,3,1024)/255.0

# model

input_1 = Input(shape=(3,1024))
layer1 = LSTM(32, activation='relu', return_sequences=True)(input_1)
layer2 = Dropout(rate=0.3)(layer1)
layer3 = LSTM(32, activation='relu')(layer2)
layer4 = Dense(512)(layer3)
output_1 = Dense(10, activation='softmax')(layer4)

model = Model(inputs=input_1, outputs=output_1)

# model = Sequential()
# model.add(Conv2D(filters=8, kernel_size=(5,5), padding='same', activation='relu', input_shape=(32,32,3)))
# model.add(Flatten())
# model.add(Dense(10, activation='sigmoid'))

# compile, fit

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train, batch_size=500, epochs=20, verbose=1)

# evaluate

loss, acc = model.evaluate(x_test, y_test, verbose=1)
print(f"loss : {loss}")
print(f"acc : {acc}")