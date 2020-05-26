from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(10, (4,4), strides=(2,2), input_shape=(10,10,1)))         # (9, 9, 10)
model.add(Conv2D(7, (3,3)))                                 # (7, 7, 7)
model.add(Conv2D(5, (2,2), padding='same'))                 # (7, 7, 5)
model.add(Conv2D(5, (2,2)))                                 # (6, 6, 5)
# model.add(Conv2D(5, (2,2), strides=2))                      # (3, 3, 5)
# model.add(Conv2D(5, (2,2), strides=2, padding='same'))      #(3, 3, 5)
model.add(MaxPooling2D(pool_size=3))                    
model.add(Flatten())                              # mutiply values to connect to Dense layer
model.add(Dense(1))

model.summary()