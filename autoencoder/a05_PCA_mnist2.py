import numpy as np 
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Activation, Input
from keras.datasets import mnist  #datasets  = 케라스에 있는 예제파일들
from sklearn.decomposition import PCA

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train[0])
# print('y_train :', y_train[0]) # 5

print(x_train.shape)  # (60000, 28, 28)
print(x_test.shape)   # (10000, 28, 28)  
print(y_train.shape)  # (60000, )디멘션 하나
print(y_test.shape)   # (10000, )



from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


#데이터 전처리 2. 정규화
x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


X = np.append(x_train, x_test, axis= 0)     # train 만 압축하면 안되니까 test도 압축하려고 append 로 묶음
print(X.shape)      # (70000, 784)


pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)
best_n_components = np.argmax(cumsum > 0.95) + 1
print(best_n_components)  # 154

pca = PCA(n_components=154)      

x_train = pca.fit_transform(x_train)     
x_test = pca.fit_transform(x_test)     
# print(pca_evr)      

print(x_train.shape, x_test.shape)
'''
pca 로 특징을 추출하였다 얼마나 추출할지 pca.explained를 사용 하였고 
argmax로 인해 153 + 1 = 154가 best n_compoenets로 판단
그리고 x_train과 x_test 둘다 트렌스폼으로 바꾸어주었다. 두개가 같아야지 차원이 맞으니까
그리고 밑에 모델링은 그대로 하였다.
'''

# 모델구성

model = Sequential()
model.add(Dense(64, input_dim=(154)))  # 아 그러면 첫번째 노드가 압축효과를 줬었겠구만..?
model.add(Dense(120, activation='relu'))
model.add(Dense(80))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax')) 

model.summary()

#3. 설명한 후 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# 걍 loss만 바꿔주면 되네?
model.fit(x_train,y_train,epochs=200,batch_size=128,validation_split=0.3,verbose=1)

#4. 평가와 예측
loss, acc = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('acc 는',acc)

predict = model.predict(x_test)
# print(predict)
print(np.argmax(predict, axis = 1))

# n = 10
# plt.figure(figsize=(20, 4))



# for i in range(n):
#     ax = plt.subplot(2, n, i+1)
#     plt.imshow(x_test[i].reshape(22, 7))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
    
#     ax = plt.subplot(2, n, i+1+n)
#     plt.imshow(decoded_imgs[i].reshape(22, 7))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

# plt.show()




# ##############################
# '''
# keras56_mnist_DNN.py 땡겨라
# input_dim = 154로 모델을 만드시오
# '''
# import numpy as np
# import matplotlib.pyplot as plt
# from keras.datasets import mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape) # (60000, 28, 28)
# print(x_test.shape) # (10000, 28, 28)
# print(y_train.shape) # (60000,)
# print(y_test.shape) # (10000,)
# # print(y_test) # [7 2 1 ... 4 5 6]
# # 전처리
# from keras.utils import np_utils
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)

# # print(len(y_test[0])) # 10
# # print(y_test) # [[0. 0. 0. ... 1. 0. 0.] ...

# x_train = (x_train/255).reshape(-1, 28*28)
# x_test = (x_test/255).reshape(-1, 28*28)

# x = np.append(x_train, x_test, axis=0)
# print(x.shape)  # (70000, 784)

# from sklearn.decomposition import PCA

# pca = PCA(n_components=154)
# pca.fit(x)
# x = pca.transform(x)
# # cumsum = np.cumsum(pca.explained_variance_ratio_)
# # print(cumsum)
# # n_components = np.argmax(cumsum >= 0.95)+1  # argmax가 어찌 돌아가는지는 찍어봐야 알듯..
# # # print(cumsum>=0.99) # True and False
# # print(n_components) # 154

# x_train = x[:60000,:]
# x_test = x[60000:,:]

# # 모델구성
# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# model = Sequential()
# model.add(Dense(100, input_dim=(154)))  # 아 그러면 첫번째 노드가 압축효과를 줬었겠구만..?
# model.add(Dense(120))
# # model.add(Dense())
# # model.add(Dense(82))
# model.add(Dense(80))
# model.add(Dense(32))
# model.add(Dense(10, activation='sigmoid')) #와... 뭐지?... 왜냐면 one-hot 인코딩 했자너 ㅠ 그니까 0/1이지 ㅠ

# model.summary()

# #3. 설명한 후 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) 
# # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# # 걍 loss만 바꿔주면 되네?
# model.fit(x_train,y_train, epochs=200, batch_size=50)  # 훨낫네.....ㅎㅎ 와 근데 미세하게 계속 올라가긴 한다잉~

# #4. 평가와 예측
# loss, acc = model.evaluate(x_test,y_test) 
# print('loss 는',loss)
# print('acc 는',acc)

# predict = model.predict(x_test)
# # print(predict)
# print(np.argmax(predict, axis = 1))
