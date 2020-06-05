import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# get data from csv
data1 = pd.read_csv('./data/csv/winequality-white.csv',
                    index_col = None,    #None
                    header = 0, 
                    sep = ';',
                    )
# print(data1.head)

# train_test_split
components = data1.iloc[:, :-1]
quality = data1.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(
    components, quality, train_size=0.9, shuffle=True
)

# find outlier

# preprocessing

# model
model = RandomForestClassifier(n_estimators=250)

# fit
model.fit(x_train, y_train)

# evaluate(score)
result = model.score(x_test, y_test) 
print(f"result : {result}")
