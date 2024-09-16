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

plt.figure()
plt.scatter(X1[:, 0], X1[:, 1], c = kmeans.labels, cmap = 'viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('PCA + KMeans Clustering with k2')
plt.savefig("./figures/pca_kmeans_k2.png")

pca = PCA(n_components = 200)
pca.fit(X)

explained_var = pca.getExplainedVariance()

plt.figure()
plt.plot(range(1, len(explained_var) + 1), explained_var)
plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.title('Explained Variance')
plt.savefig("./figures/pca_explained_variance.png")

n = 10

plt.figure()
plt.plot(range(1, n+1), pca.eig_values[:n], marker = 'o')
plt.xlabel('Number of components')
plt.ylabel('Eigenvalues fraction')
plt.title('Eigenvalues Fraction')
plt.savefig("./figures/pca_eigenvalues_fraction.png")

pca = PCA(n_components = 133)
pca.fit(X)
X2 = pca.transform(X)
X2 = np.real(X2)

wcss = []
for i in range(1, 20):
    kmeans = KMeans(k = i, n_iter = 100)
    kmeans.fit(X2)
    wcss.append(kmeans.getCost())

plt.figure()
plt.plot(range(1, 20), wcss, marker = 'o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('./figures/pca_kmeans_elbow.png')

gmm = Gmm(k = k2, n_iter = 10)
gmm.fit(X1)

clusters = np.argmax(gmm.getMembership(), axis = 1)

plt.figure()
plt.scatter(X1[:, 0], X1[:, 1], c = clusters, cmap = 'viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('PCA + GMM Clustering with k2')
plt.savefig("./figures/pca_gmm_k2.png")


AIC = []
BIC = []

for i in range(1, 11):
    gmm = Gmm(k = i, n_iter = 10)
    gmm.fit(X2)
    AIC.append(gmm.aic())
    BIC.append(gmm.bic())

plt.figure()
plt.plot(range(1, 11), AIC, marker = 'o', label = 'AIC')
plt.plot(range(1, 11), BIC, marker = 'o', label = 'BIC')
plt.xlabel('Number of clusters')
plt.ylabel('Information Criterion')
plt.legend()
plt.title('AIC and BIC for GMM')
plt.savefig("./figures/pca_gmm_aic_bic.png")

from sklearn.mixture import GaussianMixture

AIC_inbuilt = []
BIC_inbuilt = []

for i in range(1, 11):
    gmm = GaussianMixture(n_components = i, max_iter = 100)
    gmm.fit(X2)
    AIC_inbuilt.append(gmm.aic(X2))
    BIC_inbuilt.append(gmm.bic(X2))

plt.figure()
plt.plot(range(1, 11), AIC_inbuilt, marker = 'o', label = 'AIC')
plt.plot(range(1, 11), BIC_inbuilt, marker = 'o', label = 'BIC')
plt.xlabel('Number of clusters')
plt.ylabel('Information Criterion')
plt.title('AIC and BIC for GMM')
plt.legend()
plt.savefig("./figures/pca_gmm1_aic_bic_inbuilt.png")

kgmm3 = 3
gmm = Gmm(k = kgmm3, n_iter = 10)