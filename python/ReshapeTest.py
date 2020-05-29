from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
x = array([[[1,2],[2,3],[3,4]],[[2,3],[3,4],[4,5]],[[3,4],[4,5],[5,6]],[[4,5],[5,6],[6,7]]])
y = array([[4,5],[5,6],[6,7]])
# y2 = array([[4,5,6,7]])        # (1,4)
# y3 = array([[4],[5],[6],[7]])  # (4,1)

print(f"x shape : {x.shape}") # (4,3)
print(f"y shape : {y.shape}") # (4, )
# print(f"y2 shape : {y2.shape}") 
# print(f"y3 shape : {y3.shape}") 

# x = x.reshape(4,3,1)
# (4,3) > (4,3,1)  
# 4행 3열 요소에 대해 한 개씩 작업하기 위한 전처리
x = x.reshape(x.shape[0], x.shape[1] ,2) 
'''
x_shape = {batch_size, timesteps, feature}
input_shape = (timesteps, feature)
input_length = (timesteps)
input_dim = (feature)
'''

print(x.shape)
print(x)
# 2. 모델 구성
model = Sequential()
# model.add(LSTM(128, activation='relu', input_shape=(3, 1))) # 행의 수는 일단 무시
model.add(LSTM(10, input_length=3, input_dim =2))
model.add(Dense(2))

model.summary()

# 3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=1000, batch_size = 1)

x_predict = array([[5,6] [6,7] [7,8]])  # x_input.shape = (3,)
x_predict = x_predict.reshape(1,3,2)
print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)

model.summary()
# 결과 값이 만족스럽지 않은 이유 

