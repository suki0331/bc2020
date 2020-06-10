# use keras pipeline (check keras98)
# use ramdomizedsearchcv 

from sklearn.datasets import load_iris
from keras.utils import np_utils
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout

dataset = load_iris()

x = dataset.data
y = dataset.target

print(x.shape)  # (150, 4)

# one_hot_encoding
y = np_utils.to_categorical(y)

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9
)

# model
def create_hyperparameters():
    batches = [128, 256, 512]
    optimizer = ['rmsprop', 'adam', 'adadelta']
    drop = [0.1, 0.2]
    return {"model__batch_size" : batches, 
            "model__optimizer" : optimizer, 
            "model__drop": drop}

def create_model(optimizer='adam', drop=0.1): # drop=0.1 << default value
    input1 = Input(shape=(4,))
    x = Dense(256, activation='relu', name='hidden1')(input1)
    x = Dropout(rate=drop)(x)
    output1 = Dense(3, activation='softmax', name='output')(x)
    model = Model(inputs=input1, outputs=output1)
    model.compile(optimizer=optimizer, metrics=['acc'],
            loss='categorical_crossentropy')
    return model

kc = KerasClassifier(build_fn=create_model, verbose=1)

hyper = create_hyperparameters()
pipe = Pipeline([("scaler", MinMaxScaler()), ('model', kc)])

search = RandomizedSearchCV(estimator=pipe, param_distributions=hyper, cv=3)

search.fit(x, y)

acc = search.score(x_test, y_test)

print(f"search.best_estimator_ : {search.best_estimator_}")
print(f"search.best_params_ : {search.best_params_}")
print(f"search.best_index_ : {search.best_index_}")