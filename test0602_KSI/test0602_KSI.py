import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten, LSTM, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

samsung_open_1 = np.load('./data/samsung_stockprice.npy', allow_pickle=True)
hite_open = np.load('./data/hite_open.npy', allow_pickle=True)
hite_high = np.load('./data/hite_high.npy', allow_pickle=True)
hite_low = np.load('./data/hite_low.npy', allow_pickle=True)
hite_close = np.load('./data/hite_close.npy', allow_pickle=True)
hite_volume = np.load('./data/hite_volume.npy', allow_pickle=True)

# print(samsung_open_1)
# print(samsung_open_1.shape)
# print(type(samsung_open_1[0][0]))
# print(samsung_open_1[0][0])
# delete samsung 06-02 data
samsung_open = np.delete(samsung_open_1, [0][0]) #  (508,)
# print(samsung_open.shape)
# print(samsung_open.shape)
sam_0602=np.array([[51000]])
# print(sam_0602)

# # always check shapes!!
# print(samsung_open.shape)   # (508,)
# print(hite_open.shape)      # (508, 1)
# print(hite_high.shape)      # (508, 1)
# print(hite_low.shape)       # (508, 1)
# print(hite_close.shape)     # (508, 1)
# print(hite_volume.shape)    # (508, 1)


# data preprocessing

# flip data to fit time series
samsung_open = np.flip(samsung_open)
hite_open = np.flip(hite_open)
hite_high = np.flip(hite_high)
hite_low = np.flip(hite_low)
hite_close = np.flip(hite_close)
hite_volume = np.flip(hite_volume)
# print(samsung_open)



# data reshape
hite_open = hite_open.reshape(-1,)
hite_high = hite_high.reshape(-1,)
hite_low = hite_low.reshape(-1,)
hite_close = hite_close.reshape(-1,)
hite_volume = hite_volume.reshape(-1,)


print(samsung_open.shape)   # (508,)
print(hite_open.shape)      # (508,)
print(hite_high.shape)      # (508,)
print(hite_low.shape)       # (508,)
print(hite_close.shape)     # (508,)
print(hite_volume.shape)    # (508,)

aaa = hite_open-samsung_open

size=2
def split_data(seq,size):
    return_list = []
    for i in range(len(seq)-size+1):
        aa = seq[i:i+size]
        return_list.append(aa)
    return np.array(return_list)

hite_open_x_1 = split_data(hite_open ,size)
hite_open_x = hite_open_x_1[:-1]
hite_open_y = hite_open[size:]

hite_close_x_1 = split_data(hite_close ,size)
hite_close_x = hite_close_x_1[:-1]
hite_close_y = hite_close[size:]

hite_high_x_1 = split_data(hite_high ,size)
hite_high_x = hite_high_x_1[:-1]
hite_high_y = hite_high[size:]

hite_low_x_1 = split_data(hite_low ,size)
hite_low_x = hite_low_x_1[:-1]
hite_low_y = hite_low[size:]

print(hite_open_x)
print(hite_open_y)



# aaa_x_1 = split_data(aaa, size)
# aaa_x = aaa_x_1[:-1]
# aaa_y = aaa[size:]
# # hite_open_x = hite_open_x.reshape(-1,6)

# print(hite_open_x)

print(hite_open_y.shape)

input_1 = Input(shape=(2,))

layer_1_1 = Dense(1024, activation='relu')(input_1)
layer_1_2 = (Dropout(0.2))(layer_1_1)
# layer_1_3 = (Dense(256, activation='relu'))(layer_1_2)
# layer_1_4 = (Dropout(0.3))(layer_1_3)
# layer_1_5 = (Dense(512))(layer_1_4)
# layer_1_6 = (Dropout(0.2))(layer_1_5)
# layer_1_7 = (Dense(512))(layer_1_6)

input_2 = Input(shape=(2,))

layer_2_1 = Dense(1024, activation='relu')(input_2)
layer_2_2 = Dropout(0.2)(layer_2_1)

input_3 = Input(shape=(2,))

layer_3_1 = Dense(1024, activation='relu')(input_3)
layer_3_2 = Dropout(0.2)(layer_2_1)

from keras.layers.merge import concatenate
merge1 = concatenate([layer_1_2, layer_2_1, layer_3_1])
output_1 = (Dense(1, activation='relu'))(layer_1_2)


model_1 = Model(inputs=[input_1, input_2, input_3], outputs=[output_1])

# compile
model_1.compile(optimizer='adam', loss='mse')

# fit
hist = model_1.fit([hite_high_x, hite_low_x, hite_close_x], [hite_open_y], batch_size=256, epochs =3000, verbose=1, validation_split=0.1)

# evaluate
# result = model_1.evaluate(hite_open_x, hite_open_y)

# predict
y_pred = model_1.predict([hite_high_x[1:],hite_high_x[1:],hite_low_x[1:]])
print(f"y_pred : {y_pred}")

# plotting graph by using matplotlib.pyplot
loss = hist.history['loss']
val_loss = hist.history['val_loss']
# acc = hist.history['acc']
# val_acc = hist.history['val_acc']

model_1.save(filepath='.\model//test0602//test0602.h5')

plt.figure(figsize=(10,10))

# 2,1,1
# plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='best')

plt.show()
