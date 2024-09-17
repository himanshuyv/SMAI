import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("./../../")

from models.gmm.gmm import Gmm

data = pd.read_csv("./../../data/external/clustering.csv")
print(data.head())

X = data[['x','y']].to_numpy()
Y = data['color'].to_numpy()

df = pd.read_feather('./../../data/external/word-embeddings.feather')

X = df['vit'].to_numpy()
X = np.array([x for x in X])
Y = df['words'].to_numpy()

gmm = Gmm(k = 3, n_iter = 10)
gmm.fit(X)

print("Likelihood: ", gmm.getLikelihood())
print("pi: ", gmm.get_params()[0])

# ploting the clusters
gamma = gmm.getMembership()
clusters = np.argmax(gamma, axis = 1)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = clusters, cmap = 'viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('GMM Clustering')
plt.savefig("./figures/gmm1_clusters.png")


AIC = []
BIC = []

for i in range(1, 11):
    gmm = Gmm(k = i, n_iter = 10)
    gmm.fit(X)
    AIC.append(gmm.aic())
    BIC.append(gmm.bic())
    print("likelihood: ", gmm.getLikelihood())

plt.figure()
plt.plot(range(1, 11), AIC, marker = 'o', label = 'AIC')
plt.plot(range(1, 11), BIC, marker = 'o', label = 'BIC')
plt.xlabel('Number of clusters')
plt.ylabel('Information Criterion')
plt.title('AIC and BIC for GMM')
plt.legend()
plt.savefig("./figures/gmm1_aic_bic_mine.png")

# Inbuilt GMM
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components = 10, max_iter = 10)
gmm.fit(X)

print("Inbuilt GMM Likelihood: ", gmm.score(X))


AIC_inbuilt = []
BIC_inbuilt = []

for i in range(1, 11):
    gmm = GaussianMixture(n_components = i, max_iter = 100)
    gmm.fit(X)
    likelihood = gmm.score(X)
    AIC_inbuilt.append(gmm.aic(X))
    BIC_inbuilt.append(gmm.bic(X))

plt.figure()
plt.plot(range(1, 11), AIC_inbuilt, marker = 'o', label = 'AIC')
plt.plot(range(1, 11), BIC_inbuilt, marker = 'o', label = 'BIC')
plt.xlabel('Number of clusters')
plt.ylabel('Information Criterion')
plt.title('AIC and BIC for GMM')
plt.legend()
plt.savefig("./figures/gmm1_aic_bic_inbuilt.png")
