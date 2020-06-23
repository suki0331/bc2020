import numpy as np
import pickle
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8
)

model = LGBMRegressor(n_estimators=101, learning_rate=0.01)

model.fit(x_train, y_train, verbose=True, eval_metric='mse', 
eval_set=[(x_train,y_train), (x_test, y_test)],
early_stopping_rounds=20)

result = model.evals_result_
print(result)

aa = model.feature_importances_

print(aa)

for threshold in aa:
    sfm = SelectFromModel(estimator = model, threshold = threshold, prefit=True)

    sfm_x_train = sfm.transform(x_train)

    selection_model = LGBMRegressor(n_jobs=-1)
    selection_model.fit(sfm_x_train, y_train)
    # selection_model.save_model("./model/xgbsave/boston_"+str(threshold)+"sfm.dat")
    pickle.dump(model, open("./model/lgbmsave/boston.pickle.dat", "wb"))

    model2 = pickle.load(open("./model/lgbmsave/boston.pickle.dat", "rb"))


    # selection_model2 = LGBMRegressor()
    # selection_model2.load_model("./model/lgbmsave/boston_sfm.dat")
    sfm_x_test = sfm.transform(x_test)
    y_pred = selection_model.predict(sfm_x_test)
    score = r2_score(y_test, y_pred)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(threshold, sfm_x_train.shape[1], score*100.0))