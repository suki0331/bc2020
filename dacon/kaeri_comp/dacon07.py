import numpy as np
import pandas as pd

x = pd.read_csv('D:/Study/data/dacon/kaeri_comp/train_features.csv',
                sep=',',
                header=0,
                index_col=0)

print(x)

x = x.iloc[:,:].to_numpy()
print(x)
x = x.reshape(2800,375,5)

df_x = pd.DataFrame(data=x)

print(df_x)