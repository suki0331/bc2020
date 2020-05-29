from sklearn.datasets import load_iris

x, y = load_iris(return_X_y=True)

print(x.shape)
print(y.shape)