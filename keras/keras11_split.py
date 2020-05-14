import keras
import numpy as np
# get MSE from scikit_learn
from sklearn.metrics import mean_squared_error
# 1. 데이터 준비
x = np.array(range(1,101))
y = np.array(range(101,201))

x_train = x[:60]
x_val = x[60:80]
x_test = x[80:]
x_pred = np.array(range(101,121))

y_train = x[:60]
y_val = x[60:80]
y_test = x[80:]


# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 1))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))


# model.add(Dense(50))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
model.add(Dense(1))

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs = 100, batch_size=1,
            validation_data = (x_val, y_val))

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print(f"loss : {loss}, mse : {mse}")

y_pred = model.predict(x_pred)
print(f"y_predict {y_pred}")


# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(x_pred, y_pred))
print(f"RMSE : {RMSE(x_pred, y_pred)}")

# R^2 구하기
from sklearn.metrics import r2_score
r2_y_pred = r2_score(x_pred, y_pred)
print(f"R2: {r2_y_pred}")

# 과제 : R2를 음수가 아닌 0.5 이하로 줄이기. 강제로 나쁜 모델 만들기
# layers는 input과 output을 포함한 5개 이상, node는 layer당 각각 5개 이상.
# batch_size = 1
# epochs = 100 이상 (최소)
