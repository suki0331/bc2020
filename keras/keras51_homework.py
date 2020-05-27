# 2번 답
import numpy as np
y = np.array([1,2,3,4,5,1,2,3,4,5])
y = y-1

'''
from keras.utils import np_utils
y = np_utils.to_categorical(y)
print(y)
print(y.shape)
'''
# 2번 답(2)
y = np.array([1,2,3,4,5,1,2,3,4,5])
print(y.shape) # (10,)
y = y.reshape(-1,1)
print(y)

from sklearn.preprocessing import OneHotEncoder
aaa = OneHotEncoder()
aaa.fit(y)
y = aaa.transform(y).toarray()

print(y)
print(y.shape)