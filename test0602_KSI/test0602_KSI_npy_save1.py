import numpy as np
import pandas as pd

samsung = pd.read_csv("./data/csv/samsung_stockprice.csv",
                        index_col=None,
                        header=0, sep=',', encoding='EUC-KR')
print(samsung) 

# delete rows with NaN
samsung = pd.DataFrame.dropna(samsung, axis=0)
print(samsung) 

samsung_open = samsung.values
print(samsung_open)

# 2020-06-02 ~ 2018-05-04 open prices to npy
samsung_open = samsung_open[: , 1:]
# print(samsung_open)


def comma_str_to_int(list):
    for i in range(0,509):
        list[i][0] = int(list[i][0].replace(',', ''))
comma_str_to_int(samsung_open)
print(samsung_open)

np.save('./data/samsung_stockprice.npy', arr=samsung_open)
'''
print(samsung.head())  # shows first five columns
print(samsung.tail())  # shows last five columns

print("==============")
print(samsung.values)  # change pandas to numpy !!important!!

aa = samsung.values
print(type(aa)) # class 'numpy.ndarray'

# save as npy

print(x_data)
print(x_data.shape)

print(y_data)
print(y_data.shape)

np.save('./data/keras95_x_data.npy', arr=x_data)
np.save('./data/keras95_y_data.npy', arr=y_data)
'''