# from m22_XGB3_iris.py
# protect overfitting
# 1. increase the number of train data
# 2. decrease amount of features
# 3. regularization

import numpy as np
from xgboost import XGBClassifier, plot_importance
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt


dataset = load_iris()

x = dataset.data
y = dataset.target

print(x.shape)  # (506, 13)
print(y.shape)  # (506, )

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True
)

n_estimators = 100              # important 4 parameters
learning_rate = 0.01
colsample_bytree = 0.9
colsample_bylevel = 0.9


n_jobs= -1      # use all cpu threads
max_depth = 5   

parameters = [                                                                  # parameters of SVC
    {'n_estimators': [100, 200, 300], 'learning_rate' : [0.1, 0.2, 0.3],
    'max_depth':[4,5,6]},
    {"n_estimators": [90, 100, 110], 'learning_rate':[0.1, 0.001, 0.0001],
    "max_depth":[4,5,6], "colsample_bytree":[0.6, 0.9, 1]},
    {"n_estimators": [90, 110], 'learning_rate':[0.1, 0.001, 0.5],
    "max_depth":[4,5,6], "colsample_bytree":[0.6, 0.9, 1],
    "colsample_bylevel": [0.6, 0.7, 0.9]}
]
# model = XGBRegressor(max_depth=max_depth, learning_rate=learning_rate,
#                       n_estimators=n_estimators, n_jobs=n_jobs,
#                       colsample_bylevel=colsample_bylevel,
#                       colsample_bytree = colsample_bytree
# )
model = GridSearchCV(estimator=XGBClassifier(), param_grid=parameters, cv=5, n_jobs=-1)  # or cv=5

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print(f"score : {score}")

# print(f"feature importances : {model.feature_importances_}")

print(f"optimal param : {model.best_params_}")
print(f"optimal estimator : {model.best_estimator_}")
y_pred = model.predict(x_test)
# print(f"accuracy : {accuracy_score(y_test, y_pred)}")

