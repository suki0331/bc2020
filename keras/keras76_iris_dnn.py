from sklearn.datasets import load_iris
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

x, y = load_iris(return_X_y=True)

# always check data shape first
print(x.shape) # (150, 4)
print(y.shape)

# check data if you need
print(x[0])
print(y[0])

# check data if you want (this case dim = 2)
# plt.imshow(x[0])
plt.plot(x[0])
plt.plot(x[1])
plt.show()
# data preprocessing
print(max(x[-1]))
print(min(x[-1]))

# compile


# fit

