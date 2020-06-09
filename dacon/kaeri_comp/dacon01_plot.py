import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping


x_train = pd.read_csv('./data/dacon//kaeri_comp/train_features.csv',
                    header=0,
                    index_col=0,
                    sep=',')

y_train = pd.read_csv('./data/dacon//kaeri_comp/train_target.csv',
                    header=0,
                    index_col=0,
                    sep=',')

print(x_train.shape)    # (1050000, 0)
print(y_train.shape)    # (2800, 4)

# print(x_train.head(5))
# print(y_train.head(5))

x_train = x_train.iloc[:,1:].to_numpy()
print(x_train[20])
print(x_train.shape)    # (105000, 4)

x_train = x_train.reshape(2800, -1 ,4)
print(x_train[0])
print(x_train.shape)    # (2800, 375, 4)
x_train_s1 = x_train[:,:,0]
x_train_s2 = x_train[:,:,1]
x_train_s3 = x_train[:,:,2]
x_train_s4 = x_train[:,:,3]
print(x_train_s1.shape) #(2800, 375)
x_train_s1 = x_train_s1.sum(axis=1)
x_train_s2 = x_train_s2.sum(axis=1)
x_train_s3 = x_train_s3.sum(axis=1)
x_train_s4 = x_train_s4.sum(axis=1)
x_train_s = x_train_s1+x_train_s2+x_train_s3+x_train_s4

print(x_train_s1.shape)
print(x_train_s) 

y_train_mv = y_train.iloc[:,2:].to_numpy()
print(y_train_mv)
x = y_train_mv[:,0]
y = np.square(y_train_mv[:,1])

plt.scatter(x,y, alpha=0.5)
plt.xlabel('m')
plt.ylabel('v')


# plt.plot(x_train_s)
# plt.subplot(4, 1, 1)
# plt.plot(x_train_s1)


# plt.subplot(4,1,2)
# plt.plot(x_train_s2)

# plt.subplot(4,1,3)
# plt.plot(x_train_s3)

# plt.subplot(4,1,4)
# plt.plot(x_train_s4)
plt.show()