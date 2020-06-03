import numpy as np
import pandas as pd
import locale
hite = pd.read_csv("./data/csv/hite_stockprice.csv",
                        index_col=0,
                        header=0, sep=',', encoding='EUC-KR') # na_filter=0 to process NaN Values
print(hite) 

# delete rows with NaN
hite = pd.DataFrame.dropna(hite, axis=0)
print(hite)

# 2020-06-01 ~ 2018-05-04 (open, high, low, close)prices, volume to npy
hite_open = hite.values[: , 0:1]
hite_high = hite.values[: , 1:2]
hite_low = hite.values[:, 2:3]
hite_close = hite.values[:, 3:4]
hite_volume = hite.values[:, 4:5]
# print(f"hite_open : {hite_open}")
# print(f"hite_high : {hite_high}")
# print(f"hite_low : {hite_low}")
# print(f"hite_close : {hite_close}")
# print(f"hite_volume : {hite_volume}")
print(type(hite_volume[0][0]))
print(hite_volume[0][0])
# replace string with commas to integer 
def comma_str_to_int(list):
    for i in range(0,508):
        list[i][0] = int(list[i][0].replace(',', ''))

comma_str_to_int(hite_open)
comma_str_to_int(hite_high)
comma_str_to_int(hite_low)
comma_str_to_int(hite_close)
comma_str_to_int(hite_volume)


# # check the data type
# print(type(hite_volume[0][0]))
# print(hite_open)

# hite_0602_open= np.array([[39000]])
# print(hite_0602_open)

np.save('./data/hite_open.npy', arr=hite_open)
np.save('./data/hite_high.npy', arr=hite_high)
np.save('./data/hite_low.npy', arr=hite_low)
np.save('./data/hite_close.npy', arr=hite_close)
np.save('./data/hite_volume.npy', arr=hite_volume)


'''

# hite_open = pd.DataFrame(hite_stock[])
# print(hite_open)
# print(f"processed_data : {hite_open}")

print(hite.head())  # shows first five columns
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