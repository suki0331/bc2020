import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Input
from tensorflow.keras.datasets import mnist  #datasets  = 케라스에 있는 예제파일들

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train[0])
# print('y_train :', y_train[0]) # 5

print(x_train.shape)  # (60000, 28, 28)
print(x_test.shape)   # (10000, 28, 28)  
print(y_train.shape)  # (60000, )디멘션 하나
print(y_test.shape)   # (10000, )


print(x_train[0].shape)  #(28, 28)
# plt.imshow(x_train[0], 'gray')
# # plt.imshow(x_train[0])
# # plt.show()

#0 부터 9까지 분류 onehotencording
#데이터 전처리 1. 원핫인코딩
# from keras.utils import np_utils
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# print('yshape :', y_train.shape) # (60000, 10) ? 왜 10이 됬지mnist = 10으로 떨어진다.

#데이터 전처리 2. 정규화
x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255
 
print(x_train.shape)    # (60000, 784)
print(x_test.shape)     # (10000, 784)
print(y_train.shape)    # (60000,)
print(y_test.shape) # (10000,)

# print(x_test)
# print(x_train)
# print(y_test)
# print(y_train)

X = np.append(x_train, x_test, axis= 0)     # train 만 압축하면 안되니까 test도 압축하려고 append 로 묶음
print(X.shape)      # (70000, 784)

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)
best_n_components = np.argmax(cumsum >= 0.95) + 1
print(best_n_components)  # 154