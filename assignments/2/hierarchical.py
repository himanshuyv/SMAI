import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import pandas as pd

df = pd.read_feather('./../../data/external/word-embeddings.feather')

X = df['vit'].to_numpy()
X = np.array([x for x in X])
Y = df['words'].to_numpy()


linkage_methods = ['single', 'complete', 'average', 'ward', 'weighted']
dendrograms = {}

for method in linkage_methods:
    if method == 'ward':
        distance_methods = ['euclidean']
    else:
        distance_methods = ['euclidean', 'cosine']
    for distance in distance_methods:
        Z = linkage(X, method=method, metric=distance)
        dendrograms[f"{method}_{distance}"] = Z
        plt.figure()
        dendrogram(Z)
        plt.title(f"Dendrogram for {method} linkage and {distance} distance")
        plt.tight_layout()
        plt.savefig(f"./figures/dendrogram_{method}_{distance}.png")


best_method = 'complete'
best_distance = 'euclidean'


def print_clusters(kbest):
    Z = dendrograms[f"{best_method}_{best_distance}"]
    clusters = fcluster(Z, kbest, criterion='maxclust')
    clusters_list = [] 
    for i in range(1, kbest+1):
        cluster = [Y[j] for j in range(len(clusters)) if clusters[j] == i]
        clusters_list.append(cluster)
    for i in range(kbest):
        print(f"Cluster {i+1}: {clusters_list[i]}")


kbest1 = 3
kbest2 = 5

print("\nKmeans best k\n")
print_clusters(kbest1)

print("\nGmm best k\n")
print_clusters(kbest2)