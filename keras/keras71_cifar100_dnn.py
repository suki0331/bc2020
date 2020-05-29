from keras.datasets import cifar100
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, LSTM, Flatten, MaxPooling2D, Dropout, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import matplotlib.pyplot as plt 

# get data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# always check shape!!
print(x_train.shape)    # (50000, 32, 32, 3)
print(y_train.shape)    # (50000, 1)
print(x_test.shape)     # (10000, 32, 32, 3)
print(y_test.shape)     # (10000, 1)

# data preprocessing 1. one_hot encoding
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train) # (50000, 100)
y_test = np_utils.to_categorical(y_test)   # (10000, 100)

print(y_train.shape)
print(y_test.shape)

# data preprocessing 2. normalization

# check data before preprocessing
# print(x_train[0])  # data range 0~255
x_train = x_train.reshape(50000,32*32*3)/255.0
x_test = x_test.reshape(10000,32*32*3)/255.0

# Sequential Model
# model = Sequential()
# model.add(Conv2D(filters=16, kernel_size=5, padding='same', activation='relu', input_shape=(32,32,3)))
# model.add(Dropout(rate=0.2))
# model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
# model.add(Dropout(rate=0.3))
# model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
# model.add(Dropout(rate=0.4))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dense(100, activation='softmax'))

# Functional Model
input1 = Input(shape=(32*32*3,))
layer1 = Dense(1024, activation='relu')(input1) 
layer2 = Dropout(rate=0.4)(layer1)
layer3 = Dense(512, activation='relu')(layer2)
layer4 = Dropout(rate=0.3)(layer3)
layer5 = Dense(256, activation='relu')(layer4)
layer6 = Dropout(rate=0.2)(layer5)
layer9 = Dense(512, activation='relu')(layer6)
output1 = Dense(100, activation='softmax')(layer9)

model = Model(inputs=input1, outputs=output1)
# compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# fit
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
model_filepath = './model/keras71_model/{epoch:02d}-{val_loss:04f}-{acc:04f}.hdf5'
model_checkpoint = ModelCheckpoint(filepath=model_filepath, monitor='val_loss', verbose=1, save_best_only=True)
tb_hist = TensorBoard(log_dir='graph_keras71', histogram_freq=0,
                      write_graph=True, write_images=True)

hist = model.fit(x_train, y_train, batch_size=1024, epochs=30, verbose=2, validation_split=0.1, callbacks=[early_stopping, model_checkpoint, tb_hist])
# evaluate
results = model.evaluate(x_test, y_test, verbose=1)
print(f"results : {results}")

# plot graph
loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

plt.figure(figsize=(10,10))

# 2,1,1
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='best')

# 2,1,2
plt.subplot(2,1,2)
plt.plot(hist.history['acc'], marker='.', c='red', label='acc')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_acc')
plt.grid()
plt.ylim(0, 1.0) # plot graph y_range(0,1)
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='best')

plt.show()