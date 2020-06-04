from sklearn.datasets import load_iris
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1. data
data = load_iris()
x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True)

# 2. model
model = SVC()
# model = KNeighborsClassifier()
# model = RandomForestClassifier()

# 3. train
model.fit(x_train,y_train)

# 4. evaluate, predict
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print(f"result : {y_predict}")
print(f"accuracy : {acc}")

# mse = mean_squared_error(y_test, y_predict)
# r2 = r2_score(y_test, y_predict)