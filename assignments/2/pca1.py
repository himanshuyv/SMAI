import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append("./../../")

from models.pca.pca import PCA



df = pd.read_feather('./../../data/external/word-embeddings.feather')

X = df['vit'].to_numpy()
X = np.array([x for x in X])
Y = df['words'].to_numpy()


pca = PCA(n_components = 2)
pca.fit(X)

print("Before Trnasformation:", pca.checkPCA(X))
print(X.shape)
X1 = pca.transform(X)
X1 = np.real(X1)
print("After PCA")
print(X1.shape)

plt.figure()
plt.scatter(X1[:, 0], X1[:, 1])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('PCA - 2D Projection')
plt.savefig("./figures/pca1_2d.png")


pca = PCA(n_components = 3)

pca.fit(X)

print("Before Trnasformation:", pca.checkPCA(X))
print(X.shape)
X2 = pca.transform(X)
X2 = np.real(X2)
print("After PCA")
print(X2.shape)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X2[:, 0], X2[:, 1], X2[:, 2], c='b', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('PCA - 3D Projection')
plt.savefig("./figures/pca1_3d.png")