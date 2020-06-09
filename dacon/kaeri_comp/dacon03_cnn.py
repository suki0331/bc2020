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

print(x_train.shape)    # (1050000, 0)
print(y_train.shape)    # (2800, 4)

# print(x_train.head(5))
# print(y_train.head(5))


x_train_s1 = x_train.iloc[:,1].to_numpy()
x_train_s2 = x_train.iloc[:,2].to_numpy()
x_train_s3 = x_train.iloc[:,3].to_numpy()
x_train_s4 = x_train.iloc[:,4].to_numpy()

pred_s1 = pred.iloc[:,1].to_numpy()
pred_s2 = pred.iloc[:,2].to_numpy()
pred_s3 = pred.iloc[:,3].to_numpy()
pred_s4 = pred.iloc[:,4].to_numpy()

y_train_X = y_train.iloc[:,0].to_numpy()
y_train_Y = y_train.iloc[:,1].to_numpy()
y_train_M = y_train.iloc[:,2].to_numpy()
y_train_V = y_train.iloc[:,3].to_numpy()

# label_encoding
label_XY = LabelEncoder()
label_M = LabelEncoder()
label_V = LabelEncoder()

label_XY.fit(y_train_X)
label_M.fit(y_train_M)
label_V.fit(y_train_V)

y_train_X_encoded = label_XY.transform(y_train_X)
y_train_Y_encoded = label_XY.transform(y_train_Y)
y_train_M_encoded = label_M.transform(y_train_M)
y_train_V_encoded = label_V.transform(y_train_V)

print(y_train_X_encoded)
print(y_train_X_encoded.shape)  # (2800,)
print(type(y_train_X_encoded))  # numpy.ndarray

# print(label_XY.inverse_transform(y_train_X_encoded))

# one_hot_encoding
y_train_X_encoded = np_utils.to_categorical(y_train_X_encoded)
y_train_Y_encoded = np_utils.to_categorical(y_train_Y_encoded)
y_train_M_encoded = np_utils.to_categorical(y_train_M_encoded)
y_train_V_encoded = np_utils.to_categorical(y_train_V_encoded)

print(y_train_X_encoded.shape)  # (2800, 9)
print(y_train_X_encoded)

# # scaler
# scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)

x_train_s1 = x_train_s1.reshape(2800, -1 , 1)
x_train_s2 = x_train_s2.reshape(2800, -1 , 1)
x_train_s3 = x_train_s3.reshape(2800, -1 , 1)
x_train_s4 = x_train_s4.reshape(2800, -1 , 1)
pred_s1 = pred_s1.reshape(700, 375, 1)
pred_s2 = pred_s2.reshape(700, 375, 1)
pred_s3 = pred_s3.reshape(700, 375, 1)
pred_s4 = pred_s4.reshape(700, 375, 1)

# y_train = y_train.iloc[:,0:].to_numpy()
# print(y_train.shape)    # (2800, 4)

# model settings
input1 = Input(shape=(375,1))
cnn1 = Conv1D(filters=64, kernel_size=11, activation='relu')(input1)
dropout1 = Dropout(rate=0.1)(cnn1)
flatten1 = Flatten()(dropout1)
dense1 = Dense(32, activation='relu')(flatten1)

input2 = Input(shape=(375,1))
cnn2 = Conv1D(filters=64, kernel_size=11, activation='relu')(input2)
dropout2 = Dropout(rate=0.1)(cnn2)
flatten2 = Flatten()(dropout2)
dense2 = Dense(32, activation='relu')(flatten2)

input3 = Input(shape=(375,1))
cnn3 = Conv1D(filters=64, kernel_size=11, activation='relu')(input3)
dropout3 = Dropout(rate=0.1)(cnn3)
flatten3 = Flatten()(dropout3)
dense3 = Dense(32, activation='relu')(flatten3)

input4 = Input(shape=(375,1))
cnn4 = Conv1D(filters=64, kernel_size=11, activation='relu')(input4)
dropout4 = Dropout(rate=0.1)(cnn4)
flatten4 = Flatten()(dropout4)
dense4 = Dense(32, activation='relu')(flatten4)

merge1 = concatenate([dense1, dense2, dense3, dense4])

output1 = Dense(9, activation='softmax')(merge1)
output2 = Dense(9, activation='softmax')(merge1)
output3 = Dense(7, activation='softmax')(merge1)
output4 = Dense(5, activation='softmax')(merge1)

model = Model(inputs=[input1, input2, input3, input4], outputs=[output1, output2, output3, output4])

model.compile(optimizer='adam', loss='mse', metrics=['mse'])

model.fit([x_train_s1, x_train_s2, x_train_s3, x_train_s4], 
         [y_train_X_encoded, y_train_Y_encoded, y_train_M_encoded, y_train_V_encoded],
         batch_size=32, epochs=20, verbose=1, validation_split=0.15)

y_pred_X, y_pred_Y, y_pred_M, y_pred_V = model.predict([pred_s1, pred_s2, pred_s3, pred_s4])


# print(label_XY.inverse_transform([y_pred_X]))
y_pred_X = np.argmax(y_pred_X, axis=1)
y_pred_Y = np.argmax(y_pred_Y, axis=1)
y_pred_M = np.argmax(y_pred_M, axis=1)
y_pred_V = np.argmax(y_pred_V, axis=1)

y_pred_X=label_XY.inverse_transform(y_pred_X)
y_pred_Y=label_XY.inverse_transform(y_pred_Y)
y_pred_M=label_M.inverse_transform(y_pred_M)
y_pred_V=label_V.inverse_transform(y_pred_V)

blank = []
for i in range(700):
    y_input = [y_pred_X[i], y_pred_Y[i], y_pred_M[i], y_pred_V[i]]
    blank.append(y_input)
# print(blank)
index_col = np.array(range(2800,3500))
y_pred = pd.DataFrame(blank, index_col)
y_pred.to_csv('./data/dacon//kaeri_comp/sample_submission_kaeri_cnn.csv',
              header=['X','Y','M','V'],
              index=True,
              index_label='id')
