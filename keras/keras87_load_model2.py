# from keras86

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout

# load proprcessed data from mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train[0])  # image. 0~255 
# print(f"y_train[0] : {y_train[0]}")

print(x_train.shape)     # (60000, 28, 28) 60000 28*28 images
print(x_test.shape)      # (10000, 28, 28) 10000 28*28 images  
print(y_train.shape)     # (60000,)  60000 scalars
print(y_test.shape)      # (10000,)  10000 scalars

# plt.imshow(x_train[33123], 'inferno_r')  # imshow == show images
# plt.imshow(x_train[0])  
# plt.show()

# data preprocessing 1. one_hot_encoding
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)  # always check the shape!!

# data preprocessing 2. normalization
x_train = x_train.reshape(60000, 28, 28, 1)/255.
x_test = x_test.reshape(10000, 28, 28, 1)/255.
# x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255 # or /255. , /255.0
# x_test = x_test.reshape(60000, 28, 28, 1).astype('float32')/255

from keras.models import load_model
model = load_model('.\model\keras85_model\model_test01.h5')

model.add(Dropout(rate=0.1, name='dropout_3'))
model.add(Dense(15, activation='relu', name='dense_3'))
model.add(Dense(10, activation='softmax', name='dense_4'))

model.summary()


# evaluate, predict
loss_acc = model.evaluate(x_test, y_test, batch_size=1024)  # batch_size 변경 시에 loss는 변화, acc는 변화 없음
print(f"result : {loss_acc} ")

'''
# print(hist)
# print(hist.history.keys())

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print(f'acc : {acc}')
print(f'val_acc : {acc}')
print(f'loss_acc : {acc}')


# set the size of the graph
plt.figure(figsize=(10, 10))

# divide graphs to see clearly, x == epoch
# plot graph (2, 1, 1)
plt.subplot(2, 1, 1)

plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid() 
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['loss', 'val_loss'])
plt.legend(loc='upper right')

# plot graph (2, 1, 2)
plt.subplot(2, 1, 2)

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid() 
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])


plt.show()
'''