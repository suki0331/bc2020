# fram a06_ae.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import random


def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,), activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

from tensorflow.keras.datasets import mnist

train_set, test_set = mnist.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

x_train = x_train/255.
x_test = x_test/255.

print(x_train.shape, x_test.shape)  # (60000, 784) (10000, 784)

model_01 = autoencoder(hidden_layer_size=1)
model_02 = autoencoder(hidden_layer_size=2)
model_04 = autoencoder(hidden_layer_size=4)
model_08 = autoencoder(hidden_layer_size=8)
model_16 = autoencoder(hidden_layer_size=16)
model_32 = autoencoder(hidden_layer_size=32)

# model.compile(optimizer='adam', loss='mse', metrics=['acc'])                   # 32 loss = 0.0102, acc = 0.0131 나옴 ..
model_01.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])     # 32 loss = 0.0936, acc = 0.8142 나왔는데 acc 보고 판단말고 loss 를 보고 판단하여라 - 쌤
model_01.fit(x_train, x_train, epochs=10)

model_02.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model_02.fit(x_train, x_train, epochs=10)

model_04.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model_04.fit(x_train, x_train, epochs=10)

model_08.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model_08.fit(x_train, x_train, epochs=10)

model_16.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model_16.fit(x_train, x_train, epochs=10)

model_32.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model_32.fit(x_train, x_train, epochs=10)

output_01 = model_01.predict(x_test)
output_02 = model_02.predict(x_test)
output_04 = model_04.predict(x_test)
output_08 = model_08.predict(x_test)
output_16 = model_16.predict(x_test)
output_32 = model_32.predict(x_test)

# draw images
fig, axes = plt.subplots(7,5, figsize=(15,15))

random_images = random.sample(range(output_01.shape[0]), 5)
outputs = [x_test, output_01, output_02, output_04,
                   output_08, output_16, output_32]

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_images[col_num]].reshape(28,28),
                  cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()
