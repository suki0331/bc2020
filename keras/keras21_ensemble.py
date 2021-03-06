import keras
import numpy as np

# 1. 데이터 준비
x1 = np.array([range(1,101), range(311,411), range(100)])
y1 = np.array([range(711,811), range(711,811), range(100)])
# np.transpose(x) 또는 (x).T사용
x2 = np.array([range(101,201), range(411,511), range(100,200)])
y2 = np.array([range(501,601), range(711,811), range(100)])

x1 = np.transpose(x1)
y1 = np.transpose(y1)

x2 = np.transpose(x2)
y2 = np.transpose(y2)


from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    # x, y, random_state=99, shuffle=True,
    x1, y1, shuffle = False,
    train_size=0.8
)

# 2개의 set이므로 2개의 test_split 구성
from sklearn.model_selection import train_test_split
x2_train, x2_test, y2_train, y2_test = train_test_split(
    # x, y, random_state=99, shuffle=True,
    x2, y2, shuffle = False,
    train_size=0.8
)

# 2. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape=(3, ))
dense1_1 = Dense(5, activation = 'relu', name='bitking1_1')(input1)
dense1_2 = Dense(4, activation = 'relu', name='bitking1_2')(dense1_1)
dense1_3 = Dense(4, activation = 'relu', name='bitking1_3')(dense1_2)
# output1 = Dense(3)(dense1_2)

# Second model
input2 = Input(shape=(3, ))
dense2_1 = Dense(50, activation = 'relu', name='bitking2_1')(input2)
dense2_2 = Dense(24, activation = 'relu', name='bitking2_2')(dense2_1)
dense2_3 = Dense(4, activation = 'relu', name='bitking2_2')(dense2_2) # 그냥 dense2_2로 적어도 상관은 없다.
# output2 = Dense(3)(dense2_2)

# 두 모델을 엮어주는 기능 불러오기
from keras.layers.merge import concatenate
# 두개의 모델을 끝나는 부분에 layer로 합침
merge1 = concatenate([dense1_2, dense2_2])

middle1 = Dense(30)(merge1)
# 모델이름 동일하게 써도 됨
middle1 = Dense(5)(middle1)
middle1 = Dense(7)(middle1)
middle1 = Dense(7)(middle1)
middle1 = Dense(3, activation = 'relu')(middle1)
middle1 = Dense(7)(middle1)

# Devide a concatenated model into 2 models (as output)
output1 = Dense(30)(middle1)
output1_2 = Dense(7)(output1)
output1_2 = Dense(7)(output1)
output1_2 = Dense(7)(output1)
output1_3 = Dense(3)(output1_2)

output2 = Dense(40)(middle1)
output2_2 = Dense(10)(output2)
output2_2 = Dense(10)(output2)
output2_2 = Dense(10)(output2)
output2_3 = Dense(3)(output2_2)

model = Model(inputs=[input1, input2], name="model_no_1",
              outputs=[output1_3, output2_3])

model.summary()

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train],
          [y1_train, y2_train], epochs = 100, batch_size=1,
           validation_split = 0.25, verbose = 1)

# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test],
                           [y1_test, y2_test], batch_size=1)

# total loss, output1_loss, output2_loss, output1_mse, output2_mse
print(f"loss : {loss}")

[y1_pred, y2_pred] = model.predict([x1_test, x2_test])
print(f"y1_predict : {y1_pred} ")
print(f"y2_predict : {y2_pred} ")

# # RMSE 구하기
# from sklearn.metrics import mean_squared_error
# def RMSE(y1_test, y_predict):
#     return np.sqrt(mean_squared_error(y1_test, y1_pred))
# print(f"RMSE : {RMSE(y1_test, y1_pred)}")

# from sklearn.metrics import mean_squared_error
# def RMSE(y1_test, y_predict):
#     return np.sqrt(mean_squared_error(y2_test, y2_pred))
# print(f"RMSE : {RMSE(y2_test, y2_pred)}")

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
RMSE1 = RMSE(y1_test, y1_pred)
RMSE2 = RMSE(y2_test, y2_pred)
print(f"RMSE1 : {RMSE1}")
print(f"RMSE2: {RMSE2}")
print(f"RMSE : {(RMSE1+RMSE2)/2}")


# R^2 구하기
from sklearn.metrics import r2_score
r2_y1_pred = r2_score(y1_test, y1_pred)
r2_y2_pred = r2_score(y2_test, y2_pred)
print(f"R2_1: {r2_y1_pred}")
print(f"R2_2: {r2_y2_pred}")
print(f"R2: {(r2_y1_pred + r2_y2_pred)/2}")

print(model.metrics_names)