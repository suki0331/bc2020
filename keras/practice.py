import numpy as np
from keras.utils import np_utils
# from keras.optimizers import SGD
'''(10,) 와 (10,1)은 다르다 '''

# 1. 데이터
x = np.array([range(1,11)])
y = np.array([ 1, 0.1, 1, 0.1, 1, 0.1, 1, 0.1, 1, 0.1 ])

print(f"x : {x}\nx.shape : {x.shape}")
x = x.transpose()
print(f"x : {x}\nx.shape : {x.shape}")

y = np_utils.to_categorical(y)
print(f" y : {y}")
print(f" y.shape : {y.shape}")

# 2. 모델
from keras.models import Sequential
from keras.layers import Dense
# ,activation='relu'

model = Sequential()
model.add(Dense(256,activation = 'elu', input_dim = 1))
model.add(Dense(128,activation = 'elu'))
model.add(Dense(64,activation = 'elu'))
model.add(Dense(32,activation = 'elu'))
model.add(Dense(16,activation = 'elu'))
model.add(Dense(2,activation = 'sigmoid'))


''' model = Sequential()
model.add(Dense(2,input_dim=1,activation='sigmoid'))
# model.add(Dense(2,activation='sigmoid')) # activation = sigmoid? 시그모이드 함수 결과 값이 항상0,1이 나온다.
                                         # 아웃풋에 곱해주는 방법? '''
model.summary()

# 3. 컴파일(훈련준비),실행(훈련)
from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.compile(optimizer='adam',loss = 'binary_crossentropy', metrics = ['acc']) # loss에 바이너리??
# model.compile(optimizer=SGD(lr=0.2),loss = 'binary_crossentropy', metrics = ['acc'])
hist = model.fit(x,y,epochs=500)#,callbacks=[],validation_split=0.1)


from matplotlib import pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['acc'])

plt.title('keras48 loss plot')
plt.ylabel('loss & acc')
plt.xlabel('epoch')
plt.legend(['train loss','train acc'])
# plt.show()

# 4. 평가, 예측
loss,acc = model.evaluate(x,y,batch_size=1)
pred = model.predict([11,12,13,14,15,16])
print(f"pred : {pred}")
pred = np.argmax(pred,axis=1)
print(f"pred : {pred}")

print(f"loss : {loss}")
print(f"acc : {acc}")