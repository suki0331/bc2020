# binary classification

# add another metrics except 'loss'
# apply earlystopping
# plot graph

# comment the consequence below


import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

EPOCHS = 100

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size = 0.8
)

model = XGBClassifier(n_estimators=EPOCHS, learning_rate=0.1)


model.fit(x_train, y_train, verbose=True, eval_metric=['logloss','rmse','auc'],
        eval_set=[(x_train, y_train), (x_test, y_test)],
        early_stopping_rounds=2
)

result = model.evals_result()
print(result)

print(f"fi: {model.feature_importances_}")

threshold = np.sort(model.feature_importances_)

print(threshold)

for threshold in threshold:
    sfm = SelectFromModel(estimator = model, threshold = threshold, prefit=True)

    sfm_x_train = sfm.transform(x_train)

    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(sfm_x_train, y_train)

    sfm_x_test = sfm.transform(x_test)
    y_pred = selection_model.predict(sfm_x_test)
    score = r2_score(y_test, y_pred)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(threshold, sfm_x_train.shape[1], score*100.0))




# plt.plot()

# fig, ax = 