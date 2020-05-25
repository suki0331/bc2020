import numpy as np
from keras.models import Model, Input
from keras.layers import Dense, LSTM

# complete an LSTM Model.
#1. 데이터 준비
a = np.array(range(1,11))
size = 5

def split_data(seq,size):
    return_list = []
    for i in range(len(seq)-size+1):
        aa = seq[i:i+size]
        return_list.append(aa)
    return np.array(return_list)

data_split = split_data(a ,size)

print(data_split)

# def split_data_x(seq,size):
#     return_list_x = []
#     for i in range(size):
#         dsx = seq[i][0:size-1]
#         return_list_x.append(dsx)
#     return np.array(return_list_x)

# def split_data_y(seq,size):
#     return_list_y = []
#     for i in range(size):
#         dsy = seq[i][size-1]
#         return_list_y.append(dsy)
#     return np.array(return_list_y)

# data_split_x = split_data_x(data_split,size)
# data_split_y = split_data_y(data_split,size)

# print(data_split_x)
# print(data_split_y)

data_split_x = data_split[:, 0:size-1]
data_split_y = data_split[:, size-1]

data_split_x = data_split_x.reshape(data_split_x.shape[0], data_split_x.shape[1],1)
print(f"data_split_x_reshape: {data_split_x}")

# 2. 모델 불러오기
from keras.models import load_model
model_1 = load_model('./model/keras44_save.h5')
# input1 = Input(shape=(4,1))
# lstm_1 = LSTM(8)(input1)
# output1 = Dense(1)(lstm_1)

# model_1 = Model(inputs=input1, outputs = output1)

model_1.compile(optimizer='adam',loss='mse',metrics=['acc'])
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience = 10, mode = 'auto')

# 3. 실행
# 실행시킨 값 반환
hist = model_1.fit(data_split_x, data_split_y, batch_size=1, epochs=10, verbose =2, callbacks=[early_stopping])
model_1.summary()

print(hist)
print(hist.history.keys())
print(hist.history.items())

import matplotlib.pyplot as plt

plt.plot(hist.history['acc'])
# plt.plot(hist.history['mse'])
plt.title('LOSS & ACC')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.show()
'''
# 4. 예측, 평가
predict_data_x = [[7,8,9,10]]
predict_data_x= np.reshape(predict_data_x, (1,4,1)) 
predict_data_y = model_1.predict(predict_data_x)
print(f"predict_data_y : {predict_data_y}")

loss, acc = model_1.evaluate(data_split_x, data_split_y)
print(f"loss : {loss} \n acc : {acc}")
'''