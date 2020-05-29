import keras
import numpy as np
# get MSE from scikit_learn
from sklearn.metrics import mean_squared_error

# 1. 데이터 준비
x = np.array(range(1,101))
y = np.array(range(101,201))
x_pred = np.array(range(101,121))
y_true = np.array(range(201,221))

# shuffle = True is default option.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=99, shuffle=True,
    # x, y, shuffle = False,
    train_size=0.9
)

# make appropriate proportions
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, random_state=99, shuffle = True,
    # x_train, y_train, shuffle = False, 
    train_size=0.78
)

print(len(x_train))
print(x_train)
print(y_train)
print(len(x_val))
print(x_val)
print(len(x_test))
print(x_test)

'''
x_train = x[:60]
x_val = x[60:80]
x_test = x[80:]

y_train = x[:60]
y_val = x[60:80]
y_test = x[80:]
'''

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 1))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
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
def RMSE(y_true, y_predict):
    return np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE : {RMSE(y_true, y_pred)}")

# R^2 구하기
from sklearn.metrics import r2_score
r2_y_pred = r2_score(y_true, y_pred)
print(f"R2: {r2_y_pred}")

