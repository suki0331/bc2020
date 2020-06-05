# From m11_gridSearch
# SVC > RandomForest 
# cancer 적용

# KFold, cross validation score
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

# 1. data
data = load_breast_cancer()
x = data.data
y = data.target

print(x.shape)  # (569, 30) 
print(y.shape)  # (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 44)

print(x_train.shape)    # (455, 30)
print(x_test.shape)     # (114, 30)
print(y_train.shape)    # (455,)
print(y_test.shape)     # (114,)

# data reshape
# x_train = x_train.reshape(-1, 28*28)
# x_test = x_test.reshape(-1, 28*28)

# print(x_train.shape)
# print(x_test.shape)

# parameters = [                                                                  # parameters of SVC
#     {'C': [1, 10, 100, 1000], 'kernel': ["linear"]},
#     {'C': [1, 10, 100, 1000], 'kernel': ["rbf"], "gamma":[0.001, 0.0001]},
#     {'C': [1, 10, 100, 1000], 'kernel': ["sigmoid"], "gamma":[0.001, 0.0001]},
# ]

parameters = [                                                                  # parameters of SVC
    {'n_estimators': [30, 40, 50, 60, 70, 80, 90, 100, 250]},
]

kfold = KFold(n_splits=5, shuffle=True)

model = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, cv=kfold, n_jobs=-1)  # or cv=5

model.fit(x_train, y_train)

print(f"optimal param : {model.best_params_}")
print(f"optimal estimator : {model.best_estimator_}")
y_pred = model.predict(x_test)
print(f"accuracy : {accuracy_score(y_test, y_pred)}")
