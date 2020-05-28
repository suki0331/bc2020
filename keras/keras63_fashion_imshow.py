import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import fashion_mnist
# 과제 1
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# always check the shape!!!!
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

plt.imshow(x_train[0])
plt.show()
