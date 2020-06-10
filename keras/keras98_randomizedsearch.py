import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

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
x_train = x_train.reshape(-1, 28*28)/255.
x_test = x_test.reshape(-1, 28*28)/255.

# check data shape
print(f"after reshape : {x_train.shape}")   # (60000, 28, 28, 1)
print(f"after reshape : {x_test.shape}")    # (10000, 28, 28, 1)

# Dense model
def create_model(optimizer='adam', drop=0.2):
    input1 = Input(shape=(28*28,))
    x = Dense(512, activation='relu', name='hidden1')(input1)
    x = Dropout(rate=drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(input1)
    x = Dropout(rate=drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(input1)
    x = Dropout(rate=drop)(x)
    output1 = Dense(10, activation='softmax', name='output')(x)
    model1 = Model(inputs=input1, outputs=output1)
    model1.compile(optimizer=optimizer, metrics=['acc'],
                   loss='categorical_crossentropy')
    return model1

def create_hyperparameters():
    batches = [400, 500, 300]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.2, 0.1, 0.3]
    return{"batch_size" : batches, 
    "drop" : dropout}

model = KerasClassifier(build_fn=create_model, verbose=1)

hyper = create_hyperparameters()

search = RandomizedSearchCV(estimator=model, param_distributions=hyper, cv=3)
search.fit(x_train, y_train)

acc = search.score(x_test, y_test)
print(f"acc : {acc}")

print(f"search.best_estimator_ : {search.best_estimator_}")
print(f"search.best_params_ : {search.best_params_}")
print(f"search.best_index_ : {search.best_index_}")