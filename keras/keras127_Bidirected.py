from keras.datasets import imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=2000)

print(x_train.shape, x_test.shape)  # (25000,) (25000,)
print(y_train.shape, y_test.shape)  # (25000,) (25000,)

print("x_train[0] :", x_train[0])
print("y_train[0]: ", y_train[0])   # 1

print("len(x_train[0]): ", len(x_train[0]))  #리스트에는 쉐이프가 없기때문에 len 사용   # 218

# y의 카테고리 개수 출력
category = np.max(y_train) + 1
print("카테고리 : ", category)  # 2 = y값

# y의 유니크한 값들 출력
y_distri = np.unique(y_train)   # unique 찾아보기 (아래참고)
                                # 한 개 , 혹은 두 개의 1차원 ndarray 집합에 대해서 배열 내 중복된 원소 제거 후 유일한 원소를 정렬하여 반환

print("y_distri : ", y_distri)  # y_distri :  [0 1]


y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count()
print("bbb :", bbb)

'''
# bbb : 0
# 0    12500
# 1    12500
'''
print("bbb.shape : ", bbb.shape)   # bbb.shape :  (2,)



from keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(x_train, maxlen = 111, padding='pre')   # maxlen = 최댓값을 100으로 잡는다
                                                                # padding='pre' 앞에서부터 0으로 채운다
                                                                # truncating='pre' 앞에서부터 자르겠다? (공부)
x_test = pad_sequences(x_test, maxlen = 111, padding='pre')

print("len(x_train[0]): ", len(x_train[0]))  # 111
print("len(x_train[-1]):", len(x_train[-1])) # 111


print(x_train.shape, x_test.shape)  # (25000, 111) (25000, 111)

#2. 모델
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten, Conv1D, MaxPool1D, Bidirectional

model = Sequential()
model.add(Embedding(2000, 128))
model.add(Conv1D(10, 5, padding='valid', activation='relu'))
model.add(MaxPool1D(pool_size=4))
model.add(Bidirectional(LSTM(10)))    # Bidirectional 거꾸로 다시 연산을 한다 parameter = 1680
# model.add(LSTM(10))                     # parameter 840

# model.add(Flatten())
# model.add(Dense(32))
# model.add(Dense(16))
model.add(Dense(1, activation='sigmoid'))


model.summary()
'''
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(x_train, y_train, batch_size= 10, epochs=10, validation_split=0.2)

acc = model.evaluate(x_test, y_test)
print("acc : ", acc)

# 그래프
y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker='.', c='red', label='TestSet Loss')
plt.plot(y_loss, marker='.', c='blue', label='TrainSet Loss')
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
'''
# 1. imdb 검색해서 데이터 내용 확인.
# 2. word_size 전체 데이터 부분 변경해서 최상값 확인
# 3. 주간과제: groupby()의 사용법 숙지할것



