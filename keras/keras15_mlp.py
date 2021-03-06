import keras
import numpy as np

# 1. 데이터 준비
x = np.transpose(np.array([range(1,101), range(311,411), range(100)]))
y = np.transpose(np.array([range(101,201), range(711,811), range(100)]))
# np.transpose(x) 또는 (x).T사용

x_pred = np.transpose(np.array([range(301,401), range(511,611), range(100)]))
y_true = np.transpose(np.array([range(401,501), range(911,1011), range(100)]))

print(x.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=99, shuffle=True,
    # x, y, shuffle = False,
    train_size=0.8
)

print(x_train)
print(x_test)

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(40, input_dim = 3))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(40))


# model.add(Dense(50))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
model.add(Dense(3))

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs = 30, batch_size=1,
            validation_split = 0.25)


# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print(f"loss : {loss}, mse : {mse}")

y_pred = model.predict(x_pred)
print(f"y_predict : {y_pred}")

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_true, y_predict):
    return np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE : {RMSE(y_true, y_pred)}")

# R^2 구하기
from sklearn.metrics import r2_score
r2_y_pred = r2_score(y_true, y_pred)
print(f"R2: {r2_y_pred}")