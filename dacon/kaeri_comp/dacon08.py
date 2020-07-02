import numpy as np
import pandas as pd

x = pd.read_csv('D:/Study/data/dacon/kaeri_comp/train_features.csv',
                sep=',',
                header=0,
                index_col=0)
# print(x.head(5))
x_s1 = x.iloc[:,0:2].to_numpy()
x_s2 = x.iloc[:,0:3:2].to_numpy()
x_s3 = x.iloc[:,0:4:3].to_numpy()
x_s4 = x.iloc[:,0:5:4].to_numpy()

def arrival(ndarray):

    ndarray = ndarray.reshape(2800,375,2)
    # print(ndarray[0][0][1] == 0.0)

    temp = []
    # print(temp)
    for j in range(2800):
        for i in range(375):
            if ndarray[j][i][1] != 0.0:
                temp.append(round(ndarray[j][i][0],6))
                break

    temp = np.array(temp)
    # print(temp.shape)   # (2800, )
    temp = temp.reshape(2800, 1)
    # print(temp.shape)   # (2800, 1)
    return temp

x_s1_t = arrival(x_s1)
x_s2_t = arrival(x_s2)
x_s3_t = arrival(x_s3)
x_s4_t = arrival(x_s4)
# print(x_s1_t.shape)
# print(x_s1_t)
# print(x_s2_t)
# print(x_s3_t)
# print(x_s4_t)
t1 = np.concatenate((x_s1_t, x_s2_t, x_s3_t, x_s4_t), axis=1)
# print(t1)
# print(t1.shape)

y = pd.read_csv('D:/Study/data/dacon/kaeri_comp/train_target.csv',
                sep=',',
                header=0,
                index_col=0)

# y_mv= y.iloc[:, 2:4].to_numpy()
y = y.iloc[:, 0:2].to_numpy()
print(y.shape)
# print(y_mv)

# x = np.concatenate((t1, y_mv), axis=1)
print(x.shape)

from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(
    t1, y, train_size=0.8
)

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization

model = Sequential()

model.add(Dense(128, input_shape=(4,), activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='relu'))

model.compile(optimizer='adam', loss='mse')

model.fit(x_train, y_train, batch_size=32, epochs=250, verbose=2, validation_split=0.1)

model.evaluate(x_test, y_test)
