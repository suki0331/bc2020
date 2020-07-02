# from keras108_ensemble.py

# data
import numpy as np

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])

from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.utils import np_utils

# y2_train = np_utils.to_categorical(y2_train)
# print(y2_train)

model = Sequential()
model.add(Dense(10, input_shape=(1, )))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# compile, fit
model.compile(loss = ['binary_crossentropy'], optimizer='adam', 
              metrics=['acc'])
            
model.fit(x_train, y_train, batch_size=1, epochs=1000, verbose=2)

loss = model.evaluate(x_train, y_train)
print(f"loss : {loss}")

x1_pred = np.array([11,12,13,14])

y_pred = model.predict(x1_pred)
print(y_pred)

# model doesn't fit well

