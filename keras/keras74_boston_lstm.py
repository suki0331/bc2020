import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout

x, y = load_boston(return_X_y=True)

# checking data shapes
print(x.shape)
print(x.shape)

print(max(y))
print(min(y))
print(x)

# split data into train_test

x_train, x_test, y_train, y_test = train_test_split(
           x, y, test_size = 0.1
)



print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(13,)))
model.add(Dense(32))
model.add(Dropout(rate=0.3))
model.add(Dense(48))
model.add(Dropout(rate=0.1))
model.add(Dense(64))
model.add(Dropout(rate=0.25))
model.add(Dense(1))

# compile
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# fit
model.fit(x_train, y_train, batch_size=32, epochs=300, verbose=2, validation_split=0.1)

# evaluate
result = model.evaluate(x_test, y_test, verbose=1)

print(result)