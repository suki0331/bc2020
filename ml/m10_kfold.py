# KFold, cross validation score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
# warning이라는 에러를 그냥 넘어가겠다는 뜻
warnings.filterwarnings('ignore')

# 1. data
iris = pd.read_csv('./data/CSV/iris.csv', header = 0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]

print(x.shape)
print(y.shape)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 44)

kfold = KFold(n_splits=5, shuffle=True)



allAlgorithms = all_estimators(type_filter = 'classifier')

# 올 이스트메이터 안에는 사이킷런의 모든 모델이 들어가 있음

for (name, algorithm) in allAlgorithms:
    model = algorithm()


    scores = cross_val_score(model, x, y, cv=kfold)
    # model.fit(x, y)
    print(name, "acc = ", scores)


import sklearn
print(sklearn.__version__)
