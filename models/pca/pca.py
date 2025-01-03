import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.X = X
        shape = X.shape
        self.n_samples = shape[0]
        self.means = np.mean(X, axis=0)
        X = X - self.means
        self.cov = np.dot(X.T, X) / self.n_samples
        self.eig_values, self.eig_vectors = np.linalg.eig(self.cov)
        idxs = np.argsort(self.eig_values)[::-1]
        self.eig_values = self.eig_values[idxs]
        self.eig_vectors = self.eig_vectors[:, idxs]

    
    def transform(self, X):
        X = X - self.means
        return np.dot(X, self.eig_vectors[:, :self.n_components])

    def checkPCA(self, X, threshold=0.1):
        if (self.eig_vectors is None) or (self.eig_values is None):
            return False
        
        if (X.shape[1] != self.eig_vectors.shape[1]):
            return False
        
        if (self.n_components > self.eig_vectors.shape[1]):
            return False

        X = X - self.means

        X_reduced = np.dot(X, self.eig_vectors[:, :self.n_components])

        X_original = np.dot(X_reduced, self.eig_vectors[:, :self.n_components].T)

        error = np.mean(np.square(X - X_original))

        return error < threshold
    
    def getExplainedVariance(self):
        sum1 = np.cumsum(self.eig_values[:self.n_components])
        sum2 = np.sum(self.eig_values)
        return sum1 / sum2