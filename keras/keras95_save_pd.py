import numpy as np
import pandas as pd

datasets = pd.read_csv("./data/csv/iris.csv",
                        index_col=None,
                        header=0, sep=',')

print(datasets)

print(datasets.head())  # shows first five columns
print(datasets.tail())  # shows last five columns

print("==============")
print(datasets.values)  # change pandas to numpy !!important!!

aa = datasets.values
print(type(aa)) # class 'numpy.ndarray'

# save as npy
x_data = aa[: , :-2]
y_data = aa[:, -1:].reshape(-1,)

print(x_data)
print(x_data.shape)

print(y_data)
print(y_data.shape)

np.save('./data/keras95_x_data.npy', arr=x_data)
np.save('./data/keras95_y_data.npy', arr=y_data)