# from m27_eval_metric.py

import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

EPOCHS = 100
dataset = load_boston()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.8, random_state=1)

model = XGBRegressor(n_estimators=EPOCHS, learning_rate=0.1)

# model.fit(x_train, y_train, verbose=True,  eval_metric= "error",
#                 eval_set=[(x_train, y_train), (x_test, y_test)])
model.fit(x_train, y_train, verbose=True,  eval_metric=["error","logloss", "rmse"],
                eval_set=[(x_train, y_train), (x_test, y_test)])
                # early_stopping_rounds=20)
# rmse, mae, logloss, error, auc

result = model.evals_result()
print(result)

y_pred = model.predict(x_test)

r2 = r2_score(y_pred, y_test)
print(f"r2: {r2}")

epochs = len(result['validation_0']['error'])
x_axis = range(0, epochs)

# fig, ax = plt.subplots()
# ax.plot(x_axis, result['validation_0']['logloss'], label='Train')
# ax.plot(x_axis, result['validation_1']['logloss'], label='Test')
# ax.legend()
# plt.ylabel('Loss')
# plt.title('XGBoost log loss')
# plt.show()

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['rmse'], label='Train')
ax.plot(x_axis, result['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel('Loss')
plt.title('XGBoost log loss')
plt.show()