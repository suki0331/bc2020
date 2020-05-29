import keras
import numpy as np
# 1. 데이터 준비
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test =  np.array([11,12,13,14,15])
y_test =  np.array([11,12,13,14,15])
x_pred = np.array([16, 17, 18]) # predict

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 1))
model.add(Dense(3))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(50))
model.add(Dense(1))

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs = 300, batch_size=1)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print(f"loss : {loss}, acc: {acc}")

y_pred = model.predict(x_pred)
print(f"y_predict {y_pred}")