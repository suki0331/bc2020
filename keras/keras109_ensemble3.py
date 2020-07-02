# from keras108_ensemble2.py

# data
import numpy as np

x1_train = np.array([1,2,3,4,5,6,7,8,9,10])
x2_train = np.array([1,2,3,4,5,6,7,8,9,10])
y1_train = np.array([1,2,3,4,5,6,7,8,9,10])
y2_train = np.array([1,0,1,0,1,0,1,0,1,0])

from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate
from keras.utils import np_utils

# y2_train = np_utils.to_categorical(y2_train)
# print(y2_train)

input1 = Input(shape=(1,))
x1 = Dense(20, activation='relu')(input1)
x1 = Dense(20)(x1)
x1 = Dense(20)(x1)

input2 = Input(shape=(1,))
x2 = Dense(20, activation='relu')(input2)
x2 = Dense(20)(x2)
x2 = Dense(20)(x2)

merge = concatenate([x1, x2])

x3 = Dense(10)(merge)
output1 = Dense(1)(x3)  # default activation value = 'linear'

x4 = Dense(15)(merge)
x4 = Dense(15)(x4)
output2 = Dense(1, activation='sigmoid')(x4)

model = Model(inputs=[input1,input2], outputs=[output1, output2])

# compile, fit
model.compile(loss = ['mse','binary_crossentropy'], optimizer='adam', 
              loss_weights=[0.1,0.9],
              metrics=['mse','acc'])

            
model.fit([x1_train, x2_train], [y1_train, y2_train], batch_size=1, epochs=500, verbose=2)

loss = model.evaluate([x1_train, x2_train], [y1_train, y2_train])
print(f"loss : {loss}")

x1_pred = np.array([11,12,13,14])
x2_pred = np.array([11,12,13,14])
y_pred = model.predict([x1_pred, x2_pred])
print(y_pred)

# model doesn't fit well

