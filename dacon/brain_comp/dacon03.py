import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping



train = pd.read_csv('./data/dacon//brain_comp/train.csv',
                    header=0,
                    index_col=0,
                    sep=',')

test = pd.read_csv('./data/dacon//brain_comp/test.csv',
                    header=0,
                    index_col=0,
                    sep=',')

submission = pd.read_csv('./data/dacon//brain_comp/sample_submission.csv',
                    header=0,
                    index_col=0,
                    sep=',')

# print(train.head(5))
# print(test.head(5))
# print(submission.head(5))

print(train.shape)          # (10000, 75)
print(test.shape)           # (10000, 71)
print(submission.shape)     # (10000, 4)

print(train.isnull().sum()[train.isnull().sum().values > 0]) # shows number of nan values

train = train.interpolate() # linear interpolation
test = test.interpolate()   # linear interpolation
train = train.fillna(0)
test = test.fillna(0)
# print(train.isnull().sum()[train.isnull().sum().values > 0]) # shows number of nan values
# print(test.isnull().sum()[test.isnull().sum().values > 0])


# train.to_csv('./data/dacon//brain_comp/train_after_interpolation.csv',
#               header=None,
#               index=False)









































# # print(f"after interpolation : {train.isnull().sum()}")

# # df to numpy
# x = train.iloc[:, :-4].to_numpy()
# y = train.iloc[:, -4:].to_numpy()
# pred = test.to_numpy()

# # x = x.to_numpy()
# # y = y.to_numpy()
# # print(type(x))
# # print(type(y))
# # print(f"x : {x}")
# # print(f"y : {y}")
# print(x.shape)      # (10000, 71)
# print(y.shape)      # (10000, 4)
# print(pred.shape)   # (10000, 71)

# x = x.reshape(-1, 71 ,1)
# pred = pred.reshape(-1, 71 ,1)

# # train_test_split
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, shuffle=True, train_size=0.9
# )
# # # scaler
# # robust = RobustScaler()
# # x_train = robust.fit_transform(x_train)
# # x_test = robust.fit_transform(x_test)
# # pred = robust.fit_transform(pred)

# # PCA
# # pca = PCA(30)
# # x_train[1:]=pca.fit_transform(x_train[1:])
# # x_test[1:]=pca.fit_transform(x_test[1:])
# # pred[1:]=pca.fit_transform(pred[1:])

# # model
# model = Sequential()
# model.add(LSTM(256, activation='relu', input_shape=(71,1)))
# model.add(Dropout(rate=0.256))
# model.add(Dense(4, activation='relu'))

# model.compile(optimizer='adam', loss='mae', metrics=['mae','mse'])


# early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
# file_path='./data/dacon//brain_comp/{epoch:02d}-{val_loss:04f}-{mae:04f}.hdf5'
# modelcheckpoint = ModelCheckpoint(filepath=file_path, monitor='val_loss', verbose=1, save_best_only=True)
# hist = model.fit(x_train, y_train, batch_size=4, epochs=20, verbose=1, callbacks=[early_stopping], validation_split=0.05)

# result = model.evaluate(x_test, y_test, batch_size=8, verbose=1)
# print(f"result_returnvalue : {result}")

# y_pred = model.predict(np.array(pred))
# print(y_pred.shape)
# print(y_pred)

# index_col = np.array(range(10000,20000))
# y_pred = pd.DataFrame(y_pred, index_col)
# y_pred.to_csv('./data/dacon//brain_comp/sample_submission.csv',
#               header=['hhb','hbo2','ca','na'],
#               index=True,
#               index_label='id')

# # plt.subplot(2,1,1)
# # plt
