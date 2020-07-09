# 과적합 피하기 
# 2. Dropout

import numpy as np
import matplotlib.pyplot as plt
from   keras.datasets  import cifar10
from   keras.layers    import Conv2D, Dropout, Input, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation
from   keras.models    import Sequential, Model
from   keras.utils     import np_utils
from   keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

from keras.regularizers import l1, l2, l1_l2

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('x_train :',x_train.shape)
print('x_test :',x_test.shape)
print('y_train :',y_train.shape)
print('y_test :',y_test.shape)

#1. 데이터 전처리
# y_train = np_utils.to_categorical(y_train)
# y_test  = np_utils.to_categorical(y_test)

# print('x_train :',x_train.shape)
# print('x_test :',x_test.shape)
print('y_train :',y_train.shape)
print('y_test :',y_test.shape)

#2. 데이터 정규화
x_train = x_train/255
x_test  = x_test/255

print('x_train :',x_train.shape)
print('x_test :',x_test.shape)

#.3 모델구성

model = Sequential() 

model.add(Conv2D(32, kernel_size=3, padding='same',  input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(Conv2D(32, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

model.add(Conv2D(64, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(Conv2D(64, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))


model.add(Conv2D(128, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(Conv2D(128, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(Dense(10, activation='softmax'))

'''
activation 쓰기 전에 BatchNormalization 을 먼저 사용해야 된다.
'''

model.summary()

model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['acc']) #0.0001 # Adam 옆 수는 lr


#.4 모델 훈련

hist = model.fit(x_train, y_train, validation_split=0.3, epochs=20, batch_size=32, verbose=1)#, callbacks=[es,cp,tb])

#.5 평가 예측
loss,acc = model.evaluate(x_test, y_test)



print('loss : ', loss)
print('acc : ',acc)


loss_acc = loss,acc


loss     = hist.history['loss']
acc      = hist.history['acc']
val_loss = hist.history['val_loss']
val_acc  = hist.history['val_acc']



# 그래프 사이즈
plt.figure(figsize=(10, 6))

# loss 그래프
plt.subplot(2, 1, 1) 
plt.plot(hist.history['loss'],     marker='.', c='red',  label='loss')  
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss') 
plt.grid() 
plt.title('loss')       
plt.ylabel('loss')       
plt.xlabel('epoch')         
plt.legend(loc = 'upper right')

# acc 그래프
plt.subplot(2, 1, 2) 
plt.plot(hist.history['acc'],     marker='.', c='red',  label='acc')  
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_acc') 
plt.grid() 
plt.title('acc')     
plt.ylabel('acc')        
plt.xlabel('epoch')           
plt.legend(loc = 'upper right')  
plt.show()


y_pre  = model.predict(x_test)

y_pre  = np.argmax(y_pre,axis=-1)
y_test = np.argmax(y_test,axis=-1)

# print(f"y_test[0:20]:{y_test[0:20]}")
# print(f"y_pre[0:20]:{y_pre[0:20]}")


# BatchNormalization  batch 43, 46, 51, 54, 60, 63, 69
# loss :  1.5223579380035401
# acc :  0.6642000079154968

# BatchNormalization  batch 69
# loss :  1.1975092920303345
# acc :  0.7095999717712402
'''
val loss, acc가 너~~~~~무 훈련값보다 안좋음
'''
# BatchNormalization  batch 69   activation 을 selu로 바꿔봄. '''이것도 개쓰랙'''
# loss :  2.1780886863708497
# acc :  0.589900016784668

# BatchNormalization  batch 43, 46, 51, 54, 60, 63, 69   activation 을 selu로 바꿔봄
# loss :  1.7230793767929078
# acc :  0.6502000093460083