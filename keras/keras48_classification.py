import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x = np.array(range(1,11))
y = np.array([1,0,1,0,1,0,1,0,1,0])

# always pre-check data shape
print(x.shape)
print(y.shape)

# 2. 모델
model = Sequential()
model.add(Dense(100, input_dim=1, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20)) # activation value has a default option.
model.add(Dense(10, activation='linear'))
model.add(Dense(1, activation='sigmoid'))


# 3. 실행
model.compile(loss="binary_crossentropy", optimizer='adam', 
              metrics=['acc'])
model.fit(x, y, batch_size=2, epochs=100)

# 4. 평가, 예측
loss, acc= model.evaluate(x, y, batch_size=1)
print(f"loss : {loss} \n acc : {acc}")

x_pred = np.array([1,2,3])
y_pred = model.predict(x_pred)
print(f"y_pred : {y_pred}")

