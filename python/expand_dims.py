import numpy as np

a = np.array([[1,2],[3,4]])
print(a.shape)
print(f"a : {a}")
a = np.expand_dims(a, axis=1)

print(f"expanded_a.shape :  {a.shape}")

print(a)