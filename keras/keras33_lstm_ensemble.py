from numpy import array
from keras.models import Model, Input
from keras.layers import Dense, LSTM

# 1. 데이터
x1 = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
           [5,6,7], [6,7,8], [7,8,9], [8,9,10],
           [9,10,11], [10,11,12],
           [20,30,40], [30,40,50], [40,50,60]])
x2 = array([[10,20,30], [20,30,40], [30,40,50], [40,50,60],
           [50,60,70], [60,70,80], [70,80,90], [80,90,100],
           [90,100,110], [100,110,120],
           [2,3,4], [3,4,5], [4,5,6]])
x1_predict = array([55, 65, 75])  # x_input.shape = (3,)
x2_predict = array([65, 75, 85])  # x_input.shape = (3,)
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
# y2 = array([[4,5,6,7]])        # (1,4)
# y3 = array([[4],[5],[6],[7]])  # (4,1)

print(f"x1 shape : {x1.shape}") # (4,3)
print(f"x2 shape : {x2.shape}") # (4,3)
print(f"y shape : {y.shape}") # (4, )
# print(f"y2 shape : {y2.shape}") 
# print(f"y3 shape : {y3.shape}") 

# x = x.reshape(4,3,1)
# (4,3) > (4,3,1)  
# 4행 3열 요소에 대해 한 개씩 작업하기 위한 전처리
x1 = x1.reshape(x1.shape[0], x1.shape[1] ,1) 
x2 = x2.reshape(x2.shape[0], x2.shape[1] ,1) 
print(x1.shape)
print(x2.shape)

# 2. 모델 구성
input1 = Input(shape=(3,1))
lstm1 = LSTM(64, activation='relu')(input1)
dense1 = Dense(1)(lstm1)

input2 = Input(shape=(3,1))
lstm2 = LSTM(64, activation='relu')(input2)
dense2 = Dense(1)(lstm2)

from keras.layers.merge import concatenate
merge1 = concatenate([dense1, dense2])

output1 = Dense(1)(merge1)

model = Model(inputs = [input1, input2], outputs=output1)

# 3. 실행
model.compile(optimizer='adam', loss='mse')
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=15, mode='min', verbose = 1) 

# mode는 min, max, auto등이 있다.
model.fit([x1, x2], y, epochs=200, batch_size = 1, verbose= 2, callbacks=[early_stopping])
x1_predict = x1_predict.reshape(1,3,1)
x2_predict = x2_predict.reshape(1,3,1)
print(x1_predict)

y_predict = model.predict([x1_predict,x2_predict])
print(y_predict)

# 결과 값이 만족스럽지 않은 이유 

