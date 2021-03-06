import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten, MaxPooling1D, Input
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from math import sqrt

x_train = pd.read_csv('./data/dacon//kaeri_comp/train_features.csv',
                    header=0,
                    index_col=0,
                    sep=',')

y_train = pd.read_csv('./data/dacon//kaeri_comp/train_target.csv',
                    header=0,
                    index_col=0,
                    sep=',')

pred = pd.read_csv('./data/dacon//kaeri_comp/test_features.csv',
                    header=0,
                    index_col=0,
                    sep=',')

x_time_s1 = x_train.iloc[:,1].to_numpy()
x_time_s2 = x_train.iloc[:,2].to_numpy()
x_time_s3 = x_train.iloc[:,3].to_numpy()
x_time_s4 = x_train.iloc[:,4].to_numpy()
x_time_s1 = x_time_s1.reshape(2800, 375)
x_time_s2 = x_time_s2.reshape(2800, 375)
x_time_s3 = x_time_s3.reshape(2800, 375)
x_time_s4 = x_time_s4.reshape(2800, 375)


def approach_time(seq=0):
    aa = np.zeros((seq.shape[0],1))
    for j in range(seq.shape[0]):
        for i in range(seq.shape[1]):
            if seq[j,i] != 0.:
                aa[j] = i
                break
    return aa

x_approach_s1 = approach_time(x_time_s1)
x_approach_s2 = approach_time(x_time_s2)
x_approach_s3 = approach_time(x_time_s3)
x_approach_s4 = approach_time(x_time_s4)

# print(x_approach_s1)
# print(x_approach_s2)
x_approach = (x_approach_s1 + x_approach_s2 + x_approach_s3 + x_approach_s4)/4.
print(x_approach.shape) # (2800, 1)


y_location = y_train.iloc[:,:2].to_numpy()
# print(y_location.shape) 

blank = []
for i in range(y_location.shape[0]):
    dist = sqrt((y_location[i][0])**2+(y_location[i][1])**2)
    blank.append(dist)

blank = np.array(blank)

print(blank.shape)  # (2800,)

# # model

# model = Sequential()
# model.add(Dense(64, activation='relu', input_shape=(1,)))
# model.add(Dropout(0.1))
# model.add(Dense(1, activation='relu'))


# model.compile(optimizer='adam', loss='mse')

# hist = model.fit(x_approach, blank, epochs=100, verbose=1, validation_split=0)

# print(hist.history.loss)

plt.scatter(blank, x_approach)
plt.show()