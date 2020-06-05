# KFold, cross validation score
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# 1. data
iris = pd.read_csv('./data/CSV/iris.csv', header = 0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 44)

parameters = [                                                                  # parameters of SVC
    {'C': [1, 10, 100, 1000], 'kernel': ["linear"]},
    {'C': [1, 10, 100, 1000], 'kernel': ["rbf"], "gamma":[0.001, 0.0001]},
    {'C': [1, 10, 100, 1000], 'kernel': ["sigmoid"], "gamma":[0.001, 0.0001]},
]

kfold = KFold(n_splits=5, shuffle=True)

model = GridSearchCV(estimator=SVC(), param_grid=parameters, cv=kfold)  # or cv=5

model.fit(x_train, y_train)

print(f"optimal param : {model.best_params_}")
print(f"optimal estimator : {model.best_estimator_}")
y_pred = model.predict(x_test)
print(f"accuracy : {accuracy_score(y_test, y_pred)}")