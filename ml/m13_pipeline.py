import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = load_iris()

x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.85, shuffle=True, random_state=33
)

# # model
# model = SVC()

pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])

pipe.fit(x_train, y_train)
print("acc : ", pipe.score(x_test, y_test))