import numpy as np

# 1. 데이터
a = np.array(range(1,11))
size = 4 

def split_x(seq,size):
    aaa = []
    for i in range(len(a) - size + 1):
        subset = a[i : (i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print("=========================")
print(dataset)
