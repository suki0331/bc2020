# keras14_mlp를 sequential에서 함수형으로 변경
# earlystopping 적용

import keras
import numpy as np

# 1. 데이터 준비
x = np.transpose(np.array([range(1,101), range(311,411), range(100)]))
y = np.transpose(np.array([range(101,201), range(711,811), range(100)]))

# np.transpose(x) 또는 (x).T사용


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=60, shuffle=True,
    # x, y, shuffle = False,
    train_size=0.8
)

# 2. 모델 구성
from keras.models import Model
from keras.layers import Input, Dense

input1 = Input(shape=(3,))
dense1 = Dense(5, activation = 'relu', name ='first_dense')(input1)
dense2 = Dense(4)(dense1)
dense3 = Dense(5)(dense2)
dense5 = Dense(3)(dense3)

model1 = Model(inputs = input1, output = dense5, name='model_keras26')

model1.summary()

# 3. 훈련
model1.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping1 = EarlyStopping(monitor='loss', mode='auto', patience =10, verbose=1)

model1.fit(x_train, y_train, epochs = 300, batch_size = 1,
          validation_split = 0.25, verbose = 1, callbacks=[early_stopping1])

# 4. 평가, 예측
loss, mse = model1.evaluate(x_test, y_test, batch_size=1)
print(f"loss : {loss}, mse : {mse}")

y_pred = model1.predict(x_test)
print(f"y_predict : {y_pred}")

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE : {RMSE(y_test, y_pred)}")

# R^2 구하기
from sklearn.metrics import r2_score
r2_y_pred = r2_score(y_test, y_pred)
print(f"R2_score : {r2_y_pred}")
