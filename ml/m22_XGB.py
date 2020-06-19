# protect overfitting
# 1. increase the number of train data
# 2. decrease amount of features
# 3. regularization

from xgboost import XGBClassifier
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

dataset = load_boston()

x = dataset.data
y = dataset.target

print(x.shape)  # (506, 13)
print(y.shape)  # (506, )

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True
)