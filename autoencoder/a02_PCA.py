import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

dataset = load_diabetes()

print(dataset)

X = dataset.data
Y = dataset.target

print(X, Y)
print(X.shape, Y.shape) # (442, 10) (442,)

pca = PCA(n_components=5)       # 5개로 컬럼을 압축한다
x2 = pca.fit_transform((X))     # (X를 넣고 TF)
pca_evr = pca.explained_variance_ratio_ # ratio = 비율, mnist 같은 경우에 검은 배경이 차지하는 비율이 높다 하지만 우린 숫자그림이 필요하니 검은화면을 날려버린다. 
print(pca_evr)                  # [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856]
print(sum(pca_evr))             # 0.8340156689459766 - 이걸 원하는 데이터 값 700개면 700에 83프로를 인코딩해준다? 