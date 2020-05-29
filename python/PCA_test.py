import numpy as np


a=np.array([[1,2,3],[4,5,6],[7,8,9]])
b=np.array([[1,2,3],[4,5,6],[7,8,9]])
b=np.transpose(b)

c = a*b

d = b*a

print(c)
print(d)