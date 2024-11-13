import numpy as np

import sys
sys.path.append('./../../')

from models.pca.pca import PCA

class PCA_autoencoder:
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.pca = PCA(latent_dim)
    
    def fit(self, x):
        self.pca.fit(x)

    def encode(self, x):
        return self.pca.transform(x)

    def forward(self, x):
        x_encoded = self.encode(x)
        return np.dot(x_encoded, self.pca.eig_vectors[:, :self.latent_dim].T) + self.pca.means