from sklearn.datasets import load_iris
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


x, y = load_iris(return_X_y=True)

# always check data shape first
print(x.shape) # (150, 4)
print(y.shape)

# check data if you need
print(x[1])
print(y)     



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
model_filepath = './model/keras70_model/{epoch:02d}-{val_loss:04f}-{acc:04f}.hdf5'
model_checkpoint = ModelCheckpoint(filepath=model_filepath, monitor='val_loss', verbose=1, save_best_only=True)
tb_hist = TensorBoard(log_dir='graph\keras76_iris', histogram_freq=0, write_graph=True, write_images=True)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# fit
hist = model.fit(x_train, y_train, batch_size=32, epochs=300, verbose=2, validation_split=0.1, callbacks=[early_stopping, tb_hist, model_checkpoint])

# evaluate
result = model.evaluate(x_test, y_test, verbose=1)

print(result)

# # plotting graph by using matplotlib.pyplot
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
# acc = hist.history['acc']
# val_acc = hist.history['val_acc']


# plt.figure(figsize=(10,10))

# # 2,1,1
# plt.subplot(2,1,1)
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='best')

# # 2,1,2
# plt.subplot(2,1,2)
# plt.plot(hist.history['acc'], marker='.', c='red', label='acc')
# plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_acc')
# plt.grid()
# plt.ylim(0, 1.0) # plot graph y_range(0,1)
# plt.title('accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(loc='best')


# plt.show()