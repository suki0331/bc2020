# from a06_ae.py
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(tf.keras.layers.Conv2D(filters=hidden_layer_size*3, kernel_size=(3,3), padding='same', input_shape=(28,28,1), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(tf.keras.layers.Conv2D(filters=hidden_layer_size*2, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(tf.keras.layers.Conv2D(filters=hidden_layer_size*1, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(tf.keras.layers.UpSampling2D(size=(2,2)))
    model.add(tf.keras.layers.Conv2D(filters=hidden_layer_size*2, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(tf.keras.layers.UpSampling2D(size=(2,2))) 
    model.add(tf.keras.layers.Conv2D(filters=hidden_layer_size*3, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), padding='same', activation='sigmoid'))
    model.summary()
    return model

from tensorflow.keras.datasets import mnist

train_set, test_set = mnist.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

x_train = x_train/255.
x_test = x_test/255.

print(x_train.shape, x_test.shape)  # (60000, 784) (10000, 784)

model = autoencoder(hidden_layer_size=4)

# model.compile(optimizer='adam', loss='mse', metrics=['acc'])                   # 32 loss = 0.0102, acc = 0.0131 나옴 ..
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])     # 32 loss = 0.0936, acc = 0.8142 나왔는데 acc 보고 판단말고 loss 를 보고 판단하여라 - 쌤

model.fit(x_train, x_train, epochs=10)
output = model.predict(x_test)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(20, 7))

# 이미지 다섯 개를 무작위로 고른다. 
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i ==0 : 
        ax.set_ylabel("INPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토 인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap='gray')
    if i ==0 : 
        ax.set_ylabel("OUTPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()

# 154 loss: 0.0658 - acc: 0.8155

'''
중간에 hidden layer 를 잡아야됌 
우리가 최적이라고 생각한 값이 0.95 했을때 = 154개 
그럼 히든레이어에 154를 줬을때는 로스값이 떨어지긴한다. mse 보단 떨어지지만 
'''