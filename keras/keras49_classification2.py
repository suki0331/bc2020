import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x = np.array(range(1,11))
y = np.array([1,2,3,4,5,1,2,3,4,5])
y = y.reshape(-1, 1)
print(y)
print()

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
y = encoder.fit_transform(y)
print(y)
print()


# always pre-check data shape
print(x.shape)
print(y.shape)

# CLASSIFY

# 2. MODEL 

model = Sequential()
model.add(Dense(100, input_dim=1, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20)) # activation value has a default option.
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='sigmoid'))

# 3. execution
model.compile(loss="categorical_crossentropy", optimizer='adam', 
              metrics=['acc'])
model.fit(x, y, batch_size=1, epochs=100)

# 4. evaluation, prediction
loss, acc= model.evaluate(x, y, batch_size=1)
print(f"loss : {loss} \n acc : {acc}")

x_pred = np.array([1,2,3,4,5])
y_pred = model.predict(x_pred)
print(y_pred)
y_pred_argmax = np.argmax(y_pred, axis=1).reshape(-1,1)
print(y_pred_argmax+1)
