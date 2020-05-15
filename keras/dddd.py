import numpy as np

x = np.transpose(np.array([range(1,101), range(311,411), range(100)]))

print(x)
# print(x_pred)
# print(x_pred.shape)
print(x.shape)


x_pred = np.transpose(np.array([range(301,401), range(511,611), range(200)]))
y_true = np.transpose(np.array([range(401,501), range(911,1011), range(200)]))

print(x_pred)