import numpy as np
import pandas as pd

x = pd.read_csv('D:/Study/data/dacon/kaeri_comp/train_features.csv',
                sep=',',
                header=0,
                index_col=0)
# print(x.head(5))
x = x.iloc[:,1:].to_numpy()
# print(x.shape)
x = x.reshape(2800,375,4,1)
# print(x.shape)
# print(x)

y = pd.read_csv('D:/Study/data/dacon/kaeri_comp/train_target.csv',
                sep=',',
                header=0,
                index_col=0)

# print(y)
y = y.iloc[:,3].to_numpy()
# print(y.shape)
y = y.reshape(2800,1)
# print(y)
z = pd.read_csv('D:/Study/data/dacon/kaeri_comp/test_features.csv',
                sep=',',
                header=0,
                index_col=0)
z = z.iloc[:,1:].to_numpy()
# print(x.shape)
z = z.reshape(700,375,4,1)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, shuffle=True
)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, MaxPooling2D

activation = 'elu'
padding = 'valid'
model = Sequential()
nf = 19
fs = (3,1)

model.add(Conv2D(nf,fs, padding=padding, activation=activation,input_shape=(375,4,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 1)))

model.add(Conv2D(nf*2,fs, padding=padding, activation=activation))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 1)))

model.add(Conv2D(nf*4,fs, padding=padding, activation=activation))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 1)))

model.add(Conv2D(nf*8,fs, padding=padding, activation=activation))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 1)))

model.add(Conv2D(nf*16,fs, padding=padding, activation=activation))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 1)))

model.add(Conv2D(nf*32,fs, padding=padding, activation=activation))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 1)))

model.add(Flatten())
model.add(Dense(256, activation ='elu'))
model.add(Dense(128, activation ='elu'))
model.add(Dense(64, activation ='elu'))
model.add(Dense(32, activation ='elu'))
model.add(Dense(16, activation ='elu'))
model.add(Dense(8, activation ='elu'))
model.add(Dense(4))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.fit(x_train, y_train, batch_size=256, epochs=100, validation_split=0.1, verbose=2)

model.evaluate(x_test, y_test, batch_size=256, verbose=1)

z_pd = model.predict(z)
index_col = np.array(range(2800,3500))
y_pred = pd.DataFrame(z_pd, index_col)
y_pred.to_csv('./data/dacon//kaeri_comp/samplevv.csv',
              header=['M'],
              index=True,
              index_label='id')