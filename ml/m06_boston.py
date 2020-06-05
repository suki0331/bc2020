from sklearn.datasets import load_boston
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA

# 1. data
data = load_boston()
x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True
)


# 2. model
# model = SVR()
# model = KNeighborsRegressor()
model = RandomForestRegressor()
model.score()
# 3. fit
model.fit(x_train, y_train)

# predict
y_predict = model.predict(x_test)
print(f"result : {y_predict}")
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
print(f"mse : {mse} , r2 : {r2}")
