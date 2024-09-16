import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append("./../../")

from models.pca.pca import PCA
from models.kmeans.kmeans import KMeans
from models.gmm.gmm import Gmm

pca = PCA(n_components = 2)

df = pd.read_feather('./../../data/external/word-embeddings.feather')

X = df['vit'].to_numpy()
X = np.array([x for x in X])
Y = df['words'].to_numpy()


pca.fit(X)

X1 = pca.transform(X)
X1 = np.real(X1)

k2 = 4

kmeans = KMeans(k = 4, n_iter = 100)

kmeans.fit(X1)

plt.scatter(X1[:, 0], X1[:, 1], c = kmeans.labels, cmap = 'viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('PCA + KMeans Clustering with k2')
plt.show()

pca = PCA(n_components = 200)
pca.fit(X)

explained_var = pca.getExplainedVariance()
plt.plot(range(1, len(explained_var) + 1), explained_var)
plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.show()

n = 10

plt.plot(range(1, n+1), pca.eig_values[:n], marker = 'o')
plt.xlabel('Number of components')
plt.ylabel('Eigenvalues fraction')
plt.show()

pca = PCA(n_components = 5)
pca.fit(X)
X2 = pca.transform(X)

wcss = []
for i in range(1, 20):
    kmeans = KMeans(k = i, n_iter = 100)
    kmeans.fit(X2)
    wcss.append(kmeans.getCost())

plt.plot(range(1, 20), wcss, marker = 'o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

gmm = Gmm(k = k2, n_iter = 10)
gmm.fit(X1)

clusters = np.argmax(gmm.getMembership(), axis = 1)

plt.scatter(X1[:, 0], X1[:, 1], c = clusters, cmap = 'viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('PCA + GMM Clustering with k2')
plt.show()


AIC = []
BIC = []

for i in range(1, 11):
    gmm = Gmm(k = i, n_iter = 10)
    gmm.fit(X2)
    AIC.append(gmm.aic())
    BIC.append(gmm.bic())

plt.plot(range(1, 11), AIC, marker = 'o', label = 'AIC')
plt.plot(range(1, 11), BIC, marker = 'o', label = 'BIC')
plt.xlabel('Number of clusters')
plt.ylabel('Information Criterion')
plt.legend()
plt.show()


kgmm3 = 3
gmm = Gmm(k = kgmm3, n_iter = 10)