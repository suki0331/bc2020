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
from mpl_toolkits.mplot3d import Axes3D


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

y_train_int = y_train.astype(int)

# y_train = y_train.to_numpy()
print(y_train)

# for i in range(2800):
#     if y_train.loc[]


def filter(seq,num1,num2):
    blank = []
    for i in range(2800):
        if seq.loc[i][0] == num1:
            if seq.loc[i][1] == num2:
                blank.append(i)
    return blank

# print(y_train.loc[0][0]==0)

dd = filter(y_train_int, 100, 100)
print(dd)

# dd1 =y_train.iloc[dd]

# x_approach
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

approach = pd.DataFrame(data=x_approach, index=None)
print(approach)

dd1 = approach.iloc[dd]
dd2 = y_train.iloc[dd]
print(type(dd2))
print(type(dd1))

result = dd2.join(dd1)
result = result.to_numpy()
print(result)
print(type(result))


fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(result[:, 2], result[:, 3], result[:, 4],
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("dd")
ax.set_xlabel("mass")
# ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("velocity")
# ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("avg_time")
# ax.w_zaxis.set_ticklabels([])

plt.show()