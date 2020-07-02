# data
import numpy as np

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y1_train = np.array([1,2,3,4,5,6,7,8,9,10])
y2_train = np.array([1,0,1,0,1,0,1,0,1,0])

from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.utils import np_utils

# y2_train = np_utils.to_categorical(y2_train)
# print(y2_train)

input1 = Input(shape=(1,))
x1 = Dense(20, activation='relu')(input1)
x1 = Dense(20)(x1)
x1 = Dense(20)(x1)

x2 = Dense(10)(x1)
output1 = Dense(1)(x2)  # default activation value = 'linear'

x3 = Dense(15)(x1)
x3 = Dense(15)(x3)
output2 = Dense(1, activation='sigmoid')(x3)

model = Model(inputs=input1, outputs=[output1, output2])

# compile, fit
model.compile(loss = ['mse','binary_crossentropy'], optimizer='adam', 
              metrics=['mse','acc'])
            
model.fit(x_train, [y1_train, y2_train], batch_size=1, epochs=1000, verbose=2)

loss = model.evaluate(x_train, [y1_train, y2_train])
print(f"loss : {loss}")

x1_pred = np.array([11,12,13,14])
y_pred = model.predict(x1_pred)
print(y_pred)

# model doesn't fit well

