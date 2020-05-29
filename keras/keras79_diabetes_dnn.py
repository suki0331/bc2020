from sklearn.datasets import load_diabetes

x, y = load_diabetes(return_X_y=True)

print(x.shape)
print(y.shape)