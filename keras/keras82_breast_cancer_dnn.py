from sklearn.datasets import load_breast_cancer

x, y = load_breast_cancer(return_X_y=True)


print(x.shape)
print(y.shape)
