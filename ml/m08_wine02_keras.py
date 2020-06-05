import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt


# get data
data = pd.read_csv('./data/csv/winequality-white.csv',
            index_col=None,
            header=0,
            sep=';'
)           

# print(data.head(5))
x = np.array(data.iloc[:, :-1])
y = np.array(data.iloc[:, -1:])

# one_hot_encoding
y = np_utils.to_categorical(y)

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.85
)

# scaler
# robust = RobustScaler()
# x_train = robust.fit_transform(x_train)
# x_test = robust.fit_transform(x_test)
standard = StandardScaler()
x_train = standard.fit_transform(x_train)
x_test = standard.fit_transform(x_test)
# minmax = MinMaxScaler()
# x_train = minmax.fit_transform(x_train)
# x_test = minmax.fit_transform(x_test)

# # pca
# dim=3
# pca = PCA(dim)
# x_train = pca.fit_transform(x_train)
# x_test = pca.fit_transform(x_test)

# checking data shape
print(x_train.shape)    # (4163, 11)
print(y_train.shape)    # (4163, 10)
print(x_test.shape)     # (735, 11)
print(y_test.shape)     # (735, 10)

# data reshape
x_train = x_train.reshape(-1, x_train.shape[1], 1)
x_test = x_test.reshape(-1, x_test.shape[1], 1)

# # checking data
# print(x_train)

# model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(11,1)))
model.add(Dropout(rate=0.2))
model.add(Dense(10, activation='softmax'))

# compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc','mse'])

# fit
hist = model.fit(x_train, y_train, batch_size=32, epochs=250, verbose=2, validation_split=0.1)

# evaluate 
loss, acc, mse = model.evaluate(x_test, y_test, batch_size=8)
print(f"acc: {acc}")
print(f"mse: {mse}")
# predict
y_pred = model.predict(x_test)
print(y_pred[:3])
print(np.argmax(y_pred[:3], axis=1))


# def RMSE(y_test, y_pred):
#     return np.sqrt(mean_squared_error(y_test, y_pred))
# RMSE1 = RMSE(y1_test, y1_pred)
# print(f"RMSE: {RMSE2}")


plt.figure(figsize=(8,8))

plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.grid()
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['loss','val_loss'], loc='best')

plt.subplot(2,1,2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.ylim(0,1)
plt.grid()
plt.title('accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc','val_acc'])


plt.show()
