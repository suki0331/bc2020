import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings
# warning이라는 에러를 그냥 넘어가겠다는 뜻

warnings.filterwarnings('ignore')
iris = pd.read_csv('./data/CSV/iris.csv', header = 0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]
# iloc 知道几行几列就行
# loc 해더와 로케이션

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 44)

allAlgorithms = all_estimators(type_filter = 'classifier')
# all_estimator(type_ filter = 클래스 파이어 모든 모델이 들어가 있음

# 올 이스트메이터 안에는 사이킷런의 모든 모델이 들어가 있음

for (name, algorithm) in allAlgorithms:
    model = algorithm()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(name, "acc = ", accuracy_score(y_test, y_pred))
# name 과 알고리즘이 빠져나온다. 반환 값이 앞에 두개라는거 

import sklearn
print(sklearn.__version__)
