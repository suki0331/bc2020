# from m32_joblib.py

# from m31_pickle.py

# from m25_eval.py
import pickle
import numpy as np
import joblib
from xgboost import XGBClassifier, XGBRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score, accuracy_score

dataset = load_breast_cancer()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.8, random_state=1)

model = XGBClassifier(n_estimators=1000, learning_rate=0.1)

# model.fit(x_train, y_train, verbose=True,  eval_metric= "error",
#                 eval_set=[(x_train, y_train), (x_test, y_test)])
model.fit(x_train, y_train, verbose=True,  eval_metric= "rmse",
                eval_set=[(x_train, y_train), (x_test, y_test)],
                early_stopping_rounds=20)
# rmse, mae, logloss, error, auc

result = model.evals_result()
# print(result)

y_pred = model.predict(x_test)

r2 = r2_score(y_pred, y_test)
acc = accuracy_score(y_pred, y_test)
print(f"acc : {acc}")

# pickle.dump(model, open("./model/xgbsave/cancer.pickle.dat", "wb"))

# joblib.dump(model, "./model/xgbsave/cancer.joblib.dat")

model.save_model("./model/xgbsave/cancer.dat")



model2 = XGBClassifier()
model2.load_model("./model/xgbsave/cancer.dat")
# model2 = joblib.load("./model/xgbsave/cancer.joblib.dat")
# model2 = pickle.load(open("./model/xgbsave/cancer.pickle.dat", "rb"))

y_pred = model2.predict(x_test)
acc = accuracy_score(y_pred, y_test)
print(f"acc: {acc}")