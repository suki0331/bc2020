# reform by using KERAS
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras.utils import np_utils
# 1. data
x_data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_data = np.array([0, 1, 1, 0])

print(x_data.shape) # (4, 2)
print(y_data.shape) # (4, )

# one_hot_encoding
# y_data = np_utils.to_categorical(y_data)
# print(y_data)

# compile
input1 = Input(shape=(2, ))
output1 = Dense(1024, activation='relu')(input1)
output1 = Dense(1, activation='sigmoid')(output1)

model = Model(inputs=input1, outputs=output1)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# fit
model.fit(x_data, y_data, batch_size=2, epochs=5000, verbose=2)

# evaluate
loss, acc = model.evaluate(x_data, y_data, batch_size=1, verbose=1)
print(f"accuracy : {acc}")

# predict
y_pred = model.predict(x_data)
# print(f"y_pred : {y_pred}")
# y_pred = np.argmax(y_pred, axis=1)
# print(f"y_pred : {y_pred}")

'''
# 2. model
# model = SVC()
# model = KNeighborsClassifier(n_neighbors=1)


# 3. train
# model.fit(x_data, y_data)

# 4. evaluation, predict
x_test = [[0, 0], [1, 0], [0, 1], [1, 1]]
# y_predict = model.predict(x_test)

# acc = accuracy_score([0, 1, 1, 0], y_predict)
print(f"x_test, 의 예측결과 : {y_predict}")
print(f"acc : {acc}")
'''