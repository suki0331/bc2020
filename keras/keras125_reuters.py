from keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#1. 데이터

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=1000, test_split=0.2)

print(x_train.shape, x_test.shape)  #(8982,) (2246,)
print(y_train.shape, y_test.shape)  #(8982,) (2246,)

print(x_train[0])
print(y_train[0])

print(len(x_train[0]))  #리스트에는 쉐이프가 없기때문에 len 사용    # 87개

# y의 카테고리 개수 출력
category = np.max(y_train) + 1      # +1은 인덱스가 0부터 시작하기때문에
print("카테고리 : ", category)  # 46 = y값

# y의 유니크한 값들 출력
y_distri = np.unique(y_train)   # unique 찾아보기
print("y_distri : ", y_distri)
'''
y_distri :  [ 0  1  2  3  4  5  6  7  8  9 10 11 
12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 
40 41 42 43 44 45]
'''

y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count()
print("bbb :", bbb)
'''
bbb : 0
[0       55
1      432
2       74
3     3159
4     1949
5       17
6       48
7       16
8      139
9      101
10     124
11     390
12      49
13     172
14      26
15      20
16     444
17      39
18      66
19     549
20     269
21     100
22      15
23      41
24      62
25      92
26      24
27      15
28      48
29      19
30      45
31      39
32      32
33      11
34      50
35      10
36      49
37      19
38      19
39      24
40      36
41      30
42      13
43      21
44      12
45      18]
'''
print("bbb.shape : ", bbb.shape)    # bbb.shape :  (46,)


# 주간과제 : groupby() 의 사용법 숙지할 것

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, maxlen = 100, padding='pre')   # maxlen = 최댓값을 100으로 잡는다
                                                                # padding='pre' 앞에서부터 0으로 채운다
                                                                # truncating='pre' 앞에서부터 자르겠다? (공부)
x_test = pad_sequences(x_test, maxlen = 100, padding='pre')

print(len(x_train[0]))  # 100
print(len(x_train[-1])) # 100

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape)  # (8982, 100) (2246, 100)

#2. 모델
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten

model = Sequential()
model.add(Embedding(10000, 128))
model.add(LSTM(64))
model.add(Dense(46, activation='softmax'))


# model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(x_train, y_train, batch_size= 100, epochs=10, validation_split=0.2)

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


# model = Sequential()
# model.add(Embedding(1000, 128, input_length=100))
# model.add(LSTM(64))
# model.add(Dense(46, activation='softmax'))
# acc :  [1.5086167523187923, 0.6291184425354004]

# model = Sequential()
# model.add(Embedding(1000, 128))
# model.add(LSTM(64))
# model.add(Dense(46, activation='softmax'))
# acc :  [1.5008857461457155, 0.6277827024459839] 