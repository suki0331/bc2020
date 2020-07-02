# from keras107_RS_lr.py
# add activation functions as a parameter
# from keras100_hyper_lstm.py
# change LSTM to Dense layer
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Conv2D, Flatten, MaxPooling2D, LSTM
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam, Adadelta, RMSprop
from keras.activations import relu, elu
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# checking data shape
print(x_train.shape)    # (60000, 28, 28)
print(x_test.shape)     # (10000, 28, 28)
print(y_train.shape)    # (60000,)
print(y_test.shape)     # (10000,)

# one_hot_encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(f"y_train.shape : {y_train.shape}")   # (60000, 10)

# # reshape data to utilize Conv2D layer
# x_train = x_train.reshape(-1, x_train.shape[1], x_train.shape[2], 1)/255.
# x_test = x_test.reshape(-1, x_test.shape[1], x_test.shape[2], 1)/255.
x_train = x_train.reshape(-1, 2, 392)/255.
x_test = x_test.reshape(-1, 2, 392)/255.

# check data shape
print(f"x_train_after reshape : {x_train.shape}")   
print(f"x_test_after reshape : {x_test.shape}")    
# Dense model
def create_model(optimizer, dropout, learning_rate, activation):
    input1 = Input(shape=(2, 392))
    x = LSTM(256, activation=activation, name='hidden1')(input1)
    x = Dropout(rate=dropout)(x)
    x = Dense(512, activation=activation, name='hidden2')(x)
    x = Dropout(rate=dropout)(x)
    output1 = Dense(10, activation='softmax', name='output')(x)
    model = Model(inputs=input1, outputs=output1)
    model.compile(optimizer=optimizer(learning_rate=learning_rate), metrics=['acc'],
                   loss='categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = [RMSprop, Adam, Adadelta]
    dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
    learning_rate = [0.1, 0.01]
    activation = [relu, elu]
    return {"batch_size" : batches, "optimizer" : optimizers, "dropout" : dropout, "learning_rate" : learning_rate, "activation" : activation}


model = KerasClassifier(build_fn=create_model)

hyper = create_hyperparameters()
# model = KerasClassifier(build_fn=create_model)
# optimizers = [RMSprop, Adam, Adadelta]
# epochs = np.array([1, 3])
# batches = np.array([1000, 2000])
# learning_rate = [0.1, 0.01]
# param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, learning_rate = learning_rate)
search = GridSearchCV(estimator=model, param_grid=hyper, cv=3)

search.fit(x_train, y_train)

# print(search.best_estimator_)
#

