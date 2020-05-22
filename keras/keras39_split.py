from numpy import array
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
x_objective = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
           [5,6,7], [6,7,8], [7,8,9], [8,9,10],
           [9,10,11], [10,11,12],
           [2000,3000,4000], [3000,4000,5000], [4000,5000,6000],
           [100,200,300]])
x1 = array([1,2,3,4,5,6,7,8,9,10,11,12])
x2 = array([2000,3000,4000,5000,6000])
x3 = array([100,200,300])

size = 3

def split_x(seq,size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        # aaa.append([item for item in subset])
        aaa.append(subset)
    return array(aaa)

dataset1 = split_x(x1, size)
dataset2 = split_x(x2, size)
dataset3 = split_x(x3, size)
dataset = np.concatenate((dataset1,dataset2,dataset3))
print("=========================")
print(dataset)
