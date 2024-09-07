import numpy as np

class KMeans:
    def __init__(self, k = 5, n_iter = 100):
        self.k = k
        self.n_iter = n_iter
        self.prev_cost = 1e9

    def fit(self, X):
        self.X = X
        shp = self.X.shape
        self.n_samples = shp[0]
        self.n_features = shp[1]
        self.centroids = self.X[np.random.choice(self.n_samples, self.k, replace = False)]
        self.labels = np.zeros(self.n_samples)
        for _ in range(self.n_iter):
            self.labels = self.predict(self.X)
            for i in range(self.k):
                self.centroids[i] = np.mean(self.X[self.labels == i], axis = 0)
            self.cost = self.getCost()

            if (self.prev_cost - self.cost) < 1e-4:
                break
            self.prev_cost = self.cost


    def predict(self, X):
        labels = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            dist = np.linalg.norm(X[i] - self.centroids, axis = 1)
            labels[i] = np.argmin(dist)
        return labels

    def getCost(self):
        cost = 0
        for i in range(self.n_samples):
            cost += np.square(np.linalg.norm(self.X[i] - self.centroids[int(self.labels[i])]))
        return cost
