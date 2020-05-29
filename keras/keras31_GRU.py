from numpy import array
from keras.models import Sequential
from keras.layers import Dense, GRU

# 1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = array([4,5,6,7])
y2 = array([[4,5,6,7]])        # (1,4)
y3 = array([[4],[5],[6],[7]])  # (4,1)

print(f"x shape : {x.shape}") # (4,3)
print(f"y shape : {y.shape}") # (4, )
print(f"y2 shape : {y2.shape}") 
print(f"y3 shape : {y3.shape}") 

# x = x.reshape(4,3,1)
# (4,3) > (4,3,1)  
# 4행 3열 요소에 대해 한 개씩 작업하기 위한 전처리
x = x.reshape(x.shape[0], x.shape[1] ,1) 
print(x.shape)

# 2. 모델 구성
model = Sequential()
model.add(GRU(128, input_shape = (3,1))) # 행의 수는 일단 무시
model.add(Dense(1))

model.summary()

# 3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=500, batch_size = 1)

x_input = array([5, 6, 7])  # x_input.shape = (3,)
x_input = x_input.reshape(1,3,1)
print(x_input)

y_hat = model.predict(x_input)
print(y_hat)

model.summary()
# 결과 값이 만족스럽지 않은 이유 

