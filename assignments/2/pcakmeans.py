import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append("./../../")

from models.pca.pca import PCA


pca = PCA(n_components = 2)

df = pd.read_feather('./../../data/external/word-embeddings.feather')

X = df['vit'].to_numpy()
X = np.array([x for x in X])
Y = df['words'].to_numpy()


pca.fit(X)

X1 = pca.transform(X)
X1 = np.real(X1)