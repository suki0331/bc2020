import keras
import numpy as np

# 1. 데이터 준비
x = np.transpose(np.array([range(1,101), range(311,411), range(100)]))
y = np.transpose(np.array([range(101,201)]))
# np.transpose(x) 또는 (x).T사용

x_pred = np.transpose(np.array([range(301,401), range(511,611), range(100)]))
y_true = np.transpose(np.array([range(270,370)]))

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
from keras.models import Sequential, Model
from keras.layers import Dense, Input

# model = Sequential()
# change linear model to functional model

# model.add(Dense(40, input_dim = 3)

# functional model

input1 = Input(shape=(3,))

dense1 = Dense(50, activation = 'relu')(input1)
dense2 = Dense(7, activation = 'relu')(dense1)
dense3 = Dense(6, activation = 'relu')(dense2)
dense4 = Dense(5, activation = 'relu')(dense3)
dense5 = Dense(4, activation = 'relu')(dense4)
dense6 = Dense(3, activation = 'relu')(dense5)
output1 = Dense(1)(dense6)

# model.add(Dense(50))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1))

model = Model(inputs = input1, outputs = output1)
model.summary()

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs = 100, batch_size=1,
            validation_split = 0.25, verbose = 1)


# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print(f"loss : {loss}, mse : {mse}")

y_pred = model.predict(x_test)
print(f"y_predict : {y_pred}")

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_true, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE : {RMSE(y_test, y_pred)}")

# R^2 구하기
from sklearn.metrics import r2_score
r2_y_pred = r2_score(y_test, y_pred)
print(f"R2: {r2_y_pred}")