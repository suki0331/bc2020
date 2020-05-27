import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

# load proprcessed data from mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


print(x_train[0])  # image. 0~255 
print(f"y_train[0] : {y_train[0]}")

print(x_train.shape)     # (60000, 28, 28) 60000 28*28 images
print(x_test.shape)      # (10000, 28, 28) 10000 28*28 images  
print(y_train.shape)     # (60000,)  60000 scalars
print(y_test.shape)      # (10000,)  10000 scalars

plt.imshow(x_test[666], 'YlGnBu')  # imshow == show images
# plt.imshow(x_train[0])  
plt.show()

print(x_train[0].shape)  # size of an image >> always check the shape and make it consistent