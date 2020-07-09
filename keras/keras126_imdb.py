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
from keras.layers import Dense, Embedding, LSTM, Flatten, Conv1D, MaxPool1D

model = Sequential()
model.add(Embedding(2000, 128, input_length=111))
model.add(Conv1D(64, 3))
model.add(MaxPool1D(pool_size=2))
model.add(Flatten())
model.add(Dense(32))
model.add(Dense(16))
# model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))


# model.summary()

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

# 1. imdb 검색해서 데이터 내용 확인.
# 2. word_size 전체 데이터 부분 변경해서 최상값 확인
# 3. 주간과제: groupby()의 사용법 숙지할것



# model = Sequential()  batch_size= 100, epochs=10, validation_split=0.2
# model.add(Embedding(2000, 128))
# model.add(LSTM(64))
# model.add(Dense(1, activation='sigmoid'))
# acc :  [0.44498898600578307, 0.8325999975204468]'''

# model = Sequential()  batch_size= 100, epochs=10, validation_split=0.2
# model.add(Embedding(3000, 128))
# model.add(LSTM(64))
# model.add(Dense(1, activation='sigmoid'))
# acc :  [0.46255290021419526, 0.8399199843406677]


# model = Sequential()  batch_size= 100, epochs=10, validation_split=0.2
# model.add(Embedding(3000, 32))
# model.add(LSTM(64))
# model.add(Dense(128))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(1, activation='sigmoid'))
# acc :  [0.4942897580909729, 0.8231199979782104]


# model = Sequential()  batch_size= 100, epochs=10, validation_split=0.2
# model.add(Embedding(2000, 128, input_length=111))
# model.add(Conv1D(64, 3))
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))
# acc :  [0.8281749723482132, 0.8201599717140198]


# model = Sequential()
# model.add(Embedding(2000, 128, input_length=111))
# model.add(Conv1D(64, 3))
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))
# acc :  [1.9645314916086196, 0.796280026435852]


# model = Sequential()
# model.add(Embedding(2000, 128, input_length=111))
# model.add(Conv1D(64, 3))
# model.add(MaxPool1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))
# acc :  [1.552294667034149, 0.7971199750900269]


# model = Sequential()
# model.add(Embedding(2000, 128, input_length=111))
# model.add(Conv1D(64, 3))
# model.add(MaxPool1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(1, activation='sigmoid'))
# acc :  [1.3235353923606872, 0.7946400046348572]