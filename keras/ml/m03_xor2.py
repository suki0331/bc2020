from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. data
x_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_data = [0, 1, 1, 0]

# 2. model
model = SVC()

# 3. train
model.fit(x_data, y_data)

# 4. evaluation, predict
x_test = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_predict = model.predict(x_test)

acc = accuracy_score([0, 1, 1, 0], y_predict)
print(f"x_test, 의 예측결과 : {y_predict}")
print(f"acc : {acc}")