from numpy import array
from keras.models import Sequential
from keras.layers import Dense, GRU

# 1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
           [5,6,7], [6,7,8], [7,8,9], [8,9,10],
           [9,10,11], [10,11,12],
           [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
# y2 = array([[4,5,6,7]])        # (1,4)
# y3 = array([[4],[5],[6],[7]])  # (4,1)

print(f"x shape : {x.shape}") # (4,3)
print(f"y shape : {y.shape}") # (4, )
# print(f"y2 shape : {y2.shape}") 
# print(f"y3 shape : {y3.shape}") 

# x = x.reshape(4,3,1)
# (4,3) > (4,3,1)  
# 4행 3열 요소에 대해 한 개씩 작업하기 위한 전처리
x = x.reshape(x.shape[0], x.shape[1] ,1) 
print(x.shape)

# 2. 모델 구성
model = Sequential()
model.add(GRU(128, activation='relu', input_shape=(3, 1))) # 행의 수는 일단 무시
model.add(Dense(1))

model.summary()

# 3. 실행
model.compile(optimizer='adam', loss='mse')

# mode는 min, max, auto등이 있다.
model.fit(x, y, epochs=300, batch_size = 1, verbose= 2)
x_predict = array([50, 60, 70])  # x_input.shape = (3,)
x_predict = x_predict.reshape(1,3,1)
print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)

model.summary()
# 결과 값이 만족스럽지 않은 이유 

