import keras
import numpy as np

# 1. 데이터 준비 
# 2 inputs 1 output
x1 = np.array([range(1,101), range(301,401)])

# np.transpose(x) 또는 (x).T사용
y1 = np.array([range(711,811), range(711,811)])
y2 = np.array([range(101,201), range(411,511)])

############ overwrite codes from here #################

x1 = np.transpose(x1)

y1 = np.transpose(y1)
y2 = np.transpose(y2)


# express x1 > y1, y2 process with 1 train_test_split  
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, y1, y2, shuffle = False,
    train_size = 0.8
)

print(x1_train.shape)
print(y1_test.shape)


# 2. 모델 구성
from keras.models import Model
from keras.layers import Dense, Input

input1 = Input(shape=(2, ))
dense1_1 = Dense(7, activation = 'relu', name='bitking1_1')(input1)
dense1_2 = Dense(4, name='bitking1_2')(dense1_1)
dense1_3 = Dense(5, name='bitking1_3')(dense1_2)
# output1 = Dense(3)(dense1_2)

# output2 = Dense(3)(dense2_2)

    # 두 모델을 엮어주는 기능 불러오기
    # from keras.layers.merge import concatenate
    # 두개의 모델을 끝나는 부분에 layer로 합침
    # merge1 = concatenate([dense1_1])

    # middle1 = Dense(10)(merge1)
    # middle1 = Dense(10)(merge1)
    # # 모델이름 동일하게 써도 됨
    # middle1 = Dense(5)(middle1)
    # middle1 = Dense(7)(middle1)
    # middle1 = Dense(2)(middle1)
    # middle1 = Dense(7)(middle1)

# Devide a concatenated model into 2 models (as output)
output1 = Dense(5)(dense1_3)
output1 = Dense(4)(output1)
output1 = Dense(4)(output1)
output1_2 = Dense(5)(output1)
output1_3 = Dense(2)(output1_2)

output2 = Dense(5)(dense1_3)
output2 = Dense(2)(output2)
output2 = Dense(7)(output2)
output2_2 = Dense(3)(output2)
output2_3 = Dense(2)(output2_2)

model = Model(inputs=input1, name="model_no_1",
              outputs=[output1_3, output2_3])

model.summary()

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x1_train,
          [y1_train, y2_train], epochs = 130, batch_size=1,
           validation_split = 0.25, verbose = 1)

# 4. 평가, 예측
loss = model.evaluate(x1_test,
                           [y1_test, y2_test], batch_size=1)

# total loss, output1_loss, output2_loss, output1_mse, output2_mse
print(f"{loss} : {loss}")

y1_pred, y2_pred = model.predict(x1_test)
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

# 대회때는 제출한 경우 주최 측에서 알아서 추가적으로 계산하기 때문에, 평균으로 나누는 것은 예상하는 것임.
print(f"RMSE1 : {RMSE1}")

# R^2 구하기
from sklearn.metrics import r2_score
r2_y1_pred = r2_score(y1_test, y1_pred)
r2_y2_pred = r2_score(y2_test, y2_pred)
print(f"R2_1: {r2_y1_pred}")
print(f"R2_2: {r2_y2_pred}")
print(f"R2: {(r2_y1_pred+r2_y2_pred)/2}")

print(model.metrics_names)