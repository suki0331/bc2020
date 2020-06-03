import sys

import tensorflow as tf
import keras

print("tensorflow : ", tf.__version__)
print("keras : ", keras.__version__)

print()
print(sys.path)

from sklearn.datasets import load_iris
iris = load_iris()

print(iris)