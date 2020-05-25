import numpy as np
from keras.models import Model, Input
from keras.layers import Dense, LSTM

# complete an LSTM Model.
#1. 데이터 준비
a = np.array(range(1,101))
size = 5

def split_data(seq,size):
    return_list = []
    for i in range(len(seq)-size+1):
        aa = seq[i:i+size]
        return_list.append(aa)
    return np.array(return_list)

data_split = split_data(a,size)

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

print(data_split_x)
print(data_split_y)

# data_split_x = data_split_x.reshape(data_split_x.shape[0], data_split_x.shape[1],1)
# print(f"data_split_x_reshape: {data_split_x}")

# practice 2. make last 6 rows as "predict"
predict_data_x = data_split_x[len(data_split_x)-6: , :]
data_split_x = data_split_x[:len(data_split_x)-6, :]
data_split_y = data_split_y[:len(data_split_y)-6]

# practice 1. train_test_split
from sklearn.model_selection import train_test_split

x_test, x_train, y_test, y_train = train_test_split(data_split_x, data_split_y, train_size = 0.8)

# 2. 모델 구성
input1 = Input(shape=(4,))
dense_1 = Dense(7, activation='relu')(input1)
dense_2 = Dense(4)(dense_1)
output1 = Dense(1)(dense_2)

model_1 = Model(inputs=input1, outputs = output1)

model_1.compile(optimizer='adam',loss='mse',metrics=['mse'])
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience = 27, mode = 'auto', verbose = 1)

# 3. 실행
model_1.fit(data_split_x, data_split_y, batch_size=1, epochs=1000, verbose =2,
            validation_split = 0.2, callbacks=[early_stopping])
model_1.summary()

# 4. 예측, 평가
# predict_data_x = np.array([[7,8,9,10]])
# predict_data_x= np.reshape(predict_data_x, (1,4,1)) 
predict_data_y = model_1.predict(predict_data_x)
print(f"predict_data_x : {predict_data_x}")
print(f"predict_data_y : {predict_data_y}")

loss, acc = model_1.evaluate(data_split_x, data_split_y)
print(f"loss : {loss} \n acc : {acc}")