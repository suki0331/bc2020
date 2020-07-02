# data
import numpy as np
x = np.array([1,2,3,4])
y = np.array([1,2,3,4])

# model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad
# optimizer=Adam(lr=0.001)
# optimizer=RMSprop(lr=0.001)
# optimizer=SGD(lr=0.001)
# optimizer=Adadelta(lr=0.001)
optimizer=Adagrad(lr=0.001)
model = Sequential()

model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(11))
model.add(Dense(1))

model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

model.fit(x, y, epochs=10000)

loss = model.evaluate(x, y)

pred1= model.predict([3,5])
print(pred1)