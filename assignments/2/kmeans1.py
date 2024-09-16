import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('./../../')

from models.kmeans.kmeans import KMeans

# load data
df = pd.read_feather('./../../data/external/word-embeddings.feather')

X = df['vit'].to_numpy()
X = np.array([x for x in X])
Y = df['words'].to_numpy()

optimal_k = 11

# fit model
kmeans = KMeans(k = optimal_k, n_iter = 100)
kmeans.fit(X)
cost = kmeans.getCost()
print(f'Cost: {cost}')

centroids = kmeans.centroids
labels = kmeans.labels

clusters = {}
for i in range(optimal_k):
    clusters[i] = []

for i in range(len(Y)):
    clusters[labels[i]].append(Y[i])

for i in range(optimal_k):
    print(f'Cluster {i}: {clusters[i]}')

wcss = []
for i in range(1, 20):
    kmeans = KMeans(k = i, n_iter = 100)
    kmeans.fit(X)
    wcss.append(kmeans.getCost())

figure = plt.figure()
plt.plot(range(1, 20), wcss, marker = 'o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('./figures/kmeans1_elbow.png')