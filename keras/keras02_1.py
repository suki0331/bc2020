import tensorflow as tf
import keras
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

# keras02.py의 source를 끌어올것. Data만 수정한 후 model 만들 때 parameter tuning, accuracy 최댓값 나올 때까지 해보기

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([10,20,30,40,50,60,70,80,90,100])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([1010,1020,1030,1040,1050,1060,1070,1080,1090,1100])

model = Sequential()
model.add(Dense(7, input_dim =1, activation ='relu'))
model.add(Dense(3))
model.add(Dense(5, input_dim =1, activation ='relu'))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1, activation='relu'))

model.summary()
"""
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=500, batch_size=1, validation_data=(x_train, y_train))
loss, acc= model.evaluate(x_test, y_test, batch_size=1)

print("loss : ", loss)
print("acc : ", acc)

output = model.predict(x_test)
print("output : \n", output)
"""