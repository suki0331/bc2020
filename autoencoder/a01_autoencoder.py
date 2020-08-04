import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Input
from tensorflow.keras.datasets import mnist  #datasets  = 케라스에 있는 예제파일들

(x_train, y_train), (x_test, y_test) = mnist.load_data()


print(x_train.shape)  # (60000, 28, 28)
print(x_test.shape)   # (10000, 28, 28)  
print(y_train.shape)  # (60000, )디멘션 하나
print(y_test.shape)   # (10000, )


print(x_train[0].shape)  #(28, 28)




#데이터 전처리 2. 정규화
x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255
 
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# print(x_test)
# print(x_train)
# print(y_test)
# print(y_train)
# 모델구성

input_img = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_img)       # 여기에 들어가는 숫자 낮으면 낮을수록 강한 특성만 추출
decoded = Dense(784, activation='sigmoid')(encoded)     # sigmoid 를 쓰는 이유? 사진속에 특징을 추출하기 위해서 이미지속에 데이터를 0과 1로 구분을 해 둔상태에서 1이 비교적 특징이 있으니 sigmoid사용

autoencoder = Model(inputs= input_img, outputs= decoded)

autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, validation_split=0.2)

decoded_imgs = autoencoder.predict(x_test)


n = 10
plt.figure(figsize=(20, 4))



for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()