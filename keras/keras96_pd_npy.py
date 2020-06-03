# load keras95 to complete model

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import np_utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# load npy data
x = np.load('./data/keras95_x_data.npy')
y = np.load('./data/keras95_y_data.npy')
# print(x_train[0])  # image. 0~255 
# print(f"y_train[0] : {y_train[0]}")

# data preprocessing
pca = PCA(1)
x = pca.fit_transform(x)

# print("pca")
# print(x)

scaler = RobustScaler()
x = scaler.fit_transform(x)

# split data into train_test

x_train, x_test, y_train, y_test = train_test_split(
           x, y, test_size = 0.1
)

# result data preprocessing one_hot_encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# # draw data if you want (this case dim = 2)
# # plt.imshow(x[0])
# plt.plot(x[0])
# plt.plot(x[1])
# plt.show()
# # data preprocessing
# print(max(x[-1]))
# print(min(x[-1]))


model = Sequential()
model.add(Dense(16, input_shape=(1,), activation='relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(32))
model.add(Dropout(rate=0.3))
model.add(Dense(16))
model.add(Dropout(rate=0.25))
model.add(Dense(3, activation='softmax'))

# compile
early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
model_filepath = './model/keras96_model/{epoch:02d}-{val_loss:04f}-{acc:04f}.hdf5'
model_checkpoint = ModelCheckpoint(filepath=model_filepath, monitor='val_loss', verbose=1, save_best_only=True)
tb_hist = TensorBoard(log_dir='graph\keras96_iris', histogram_freq=0, write_graph=True, write_images=True)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# fit
hist = model.fit(x_train, y_train, batch_size=32, epochs=300, verbose=2, validation_split=0.1, callbacks=[early_stopping, tb_hist, model_checkpoint])

# evaluate
result = model.evaluate(x_test, y_test, verbose=1)

print(result)