import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

x, y = load_boston(return_X_y=True)

# checking data shapes
print(x.shape)  # (506, 13)
print(y.shape)  # (506, )

print(max(y))
print(min(y))
print(x)

# data preprocessing
# pca = PCA()
# x = pca.fit_transform(x)

# print("pca")
# print(x)

scaler = RobustScaler()
x = scaler.fit_transform(x)

# reshape data after scaling
x = x.reshape(-1,13,1,1)

# print("rbsc")
# print(x)


# split data into train_test

x_train, x_test, y_train, y_test = train_test_split(
           x, y, test_size = 0.1
)


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu', input_shape=(13,1,1)))
model.add(Flatten())
model.add(Dense(16))
model.add(Dropout(rate=0.3))
model.add(Dense(32))
model.add(Dropout(rate=0.3))
model.add(Dense(16))
model.add(Dropout(rate=0.25))
model.add(Dense(1))

# compile
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# fit
model.fit(x_train, y_train, batch_size=32, epochs=300, verbose=2, validation_split=0.1)

# evaluate
result = model.evaluate(x_test, y_test, verbose=1)

print(result)