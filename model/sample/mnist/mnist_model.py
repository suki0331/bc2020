import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# always check data!!

print(x_train.shape)    # (60000, 28, 28)
print(y_train.shape)    # (60000,)
print(x_test.shape)     # (10000, 28, 28)
print(y_test.shape)     # (10000,)
# print(np.array([x_test[15]]).shape)
# print(x_train[0])
# print(y_train[0])
# print(x_test[15])
# print(y_test[0])

# reshape data
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
# data preprocessing

# applying scaler
# scaler = MinMaxScaler()
# x = scaler.fit_transform(x)

# applying pca
# pca = PCA()
# x = pca.fit_transform(x)

# one_hot_encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)    # (60000, 10)

# model
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=5 ,activation='relu', input_shape=(28,28,1)))
model.add(Dropout(rate=0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# fit
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
tensor_board = TensorBoard(log_dir='graph', histogram_freq=0,
                      write_graph=True, write_images=True)
model_filepath='.\\model\\sample\\mnist\\'
model_checkpoint = ModelCheckpoint(filepath=model_filepath+'{epoch:02d}-{val_loss:04f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)
hist = model.fit(x_train, y_train, batch_size=1024, epochs=6, verbose=2, validation_split=0.1 ,callbacks=[early_stopping,model_checkpoint, tensor_board])

model.save(filepath=model_filepath +'model_save.h5')

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']
# evaluate
result = model.evaluate(x_test, y_test, batch_size=32, verbose=1)
print(f"result : {result}")

# predict 
x_test[15] = np.array(x_test[15]).reshape(1,28,28,1)
y_pred = model.predict(np.array([x_test[15]]))
print(f"y_pred : {y_pred}")
y_pred = np.argmax(y_pred)
print(f"argmax.y_pred : {y_pred}")

# # RMSE 
# from sklearn.metrics import mean_squared_error
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_pred))
# print(f"RMSE : {RMSE(y_test, y_pred)}")

# # R^2 
# from sklearn.metrics import r2_score
# r2_y_pred = r2_score(y_test, y_pred)
# print(f"R2: {r2_y_pred}")

# plotting graph
plt.figure(figsize=(10, 10))

plt.subplot(2, 1, 1)

plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid() 
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['loss', 'val_loss'])
plt.legend(loc='upper right')


plt.subplot(2, 1, 2)

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid() 
plt.ylim(0, 1.0) # plot graph y_range(0,1)
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()