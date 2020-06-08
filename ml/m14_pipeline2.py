# from m13

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV

data = load_iris()

x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.85, shuffle=True, random_state=33
)

parameters = [
    {"svm__C":[1,10,100,1000], "svm__kernel":['linear']},
    {"svm__C":[1,10,100,1000], "svm__kernel":['rbf']},
    {"svm__C":[1,10,100,1000], "svm__kernel":['sigmoid'], "svm__gamma":[0.001,0.0001] },
]

# # model
# model = SVC()

pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])

model = RandomizedSearchCV(pipe, parameters, cv=5)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(f"optimal parameter: {model.best_params_}")
print(f"acc : {acc}")
