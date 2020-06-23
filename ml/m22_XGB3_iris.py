# from m22_XGB.py
# protect overfitting
# 1. increase the number of train data
# 2. decrease amount of features
# 3. regularization

import numpy as np
from xgboost import XGBClassifier, plot_importance
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
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

model = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate,
                      n_estimators=n_estimators, n_jobs=n_jobs,
                      colsample_bylevel=colsample_bylevel,
                      colsample_bytree = colsample_bytree
)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print(f"score : {score}")

print(f"feature importances : {model.feature_importances_}")


plot_importance(model)
plt.show()
