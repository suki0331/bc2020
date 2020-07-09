import numpy as np
from keras.datasets import cifar10
from keras.applications import VGG16, InceptionV3
from keras.utils import np_utils
from keras.models import Input, Model
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import SGD
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)    # (50000, 32, 32, 3)
print(y_train.shape)    # (50000, 1)
print(x_test.shape)     # (10000, 32, 32, 3)
print(y_test.shape)     # (10000, 1)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)    # (50000, 10)

input_tensor = Input(shape=(32, 32, 3))
inceptionv3 = InceptionV3(include_top=False, weights='imagenet', input_tensor=input_tensor)

top_model = inceptionv3.output
top_model = Flatten(input_shape=inceptionv3.output_shape[1:])(top_model)
top_model = Dense(256, activation = 'sigmoid')(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(10, activation='softmax')(top_model)

model = Model(inputs=inceptionv3.input, outputs=top_model)

for layer in model.layers[:19]:
    layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=1e-4, momentum=0.9),
              metrics = ['accuracy'])
            
model.fit(x_train, y_train, validation_data=(x_test,y_test), batch_size=32, epochs=5)


scores = model.evaluate(x_test, y_test, verbose=1)
print(f"test loss : {scores[0]}")
print(f"test acc : {scores[1]}")


for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[i])
plt.suptitle("data", fontsize=16)
plt.show()

pred = np.argmax(model.predict(x_test[0:10]), axis=1)
print(pred)