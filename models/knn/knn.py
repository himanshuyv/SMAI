import numpy as np

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        
    def predict(self, x_test, distance_metric='euclidean'):
        y_pred = []
        for x in x_test.values:
            y_pred.append(self.predict_one(x, distance_metric))
        return np.array(y_pred)
    
    def predict_one(self, x, distance_metric):
        if distance_metric == 'euclidean':
            distances = np.sqrt(np.sum((self.x_train - x) ** 2, axis=1))
        elif distance_metric == 'manhattan':
            distances = np.sum(np.abs(self.x_train - x), axis=1)
        elif distance_metric == 'cosine':
            distances = np.dot(self.x_train, x) / (np.linalg.norm(self.x_train) * np.linalg.norm(x))
        nearest_indices = np.argsort(distances)[:self.k]
        nearest_labels = self.y_train.iloc[nearest_indices]
        return nearest_labels.mode().values[0]
