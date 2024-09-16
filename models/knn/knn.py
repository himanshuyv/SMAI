import numpy as np

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, x_train, y_train):
        if (type(x_train) == np.ndarray):
            self.x_train = x_train
        else:
            self.x_train = x_train.to_numpy()
        if (type(y_train) == np.ndarray):
            self.y_train = y_train
        else:
            self.y_train = y_train.to_numpy()
        self.norm_x_train = np.linalg.norm(self.x_train, axis=1)
        
    def predict(self, x_test, distance_metric='euclidean'):
        y_pred = []
        if (type(x_test) != np.ndarray):
            x_test = x_test.to_numpy()
                    
        for x in x_test:
            y_pred.append(self.predict_one(x, distance_metric))
        return np.array(y_pred)
    
    def predict_one(self, x, distance_metric):
        if distance_metric == 'euclidean':
            distances = np.sqrt(np.sum((self.x_train - x) ** 2, axis=1))
        elif distance_metric == 'manhattan':
            distances = np.sum(np.abs(self.x_train - x), axis=1)
        elif distance_metric == 'cosine':
            # For this looked up online resourses for how to calculate cosine distance efficiently
            dot_product = np.dot(self.x_train, x)
            norm_x = np.linalg.norm(x)
            cosine_similarity = dot_product / (norm_x * self.norm_x_train)
            distances = 1 - cosine_similarity

        # For this also looked up online resourses for how to get k nearest neighbours efficiently  
        nearest_indices = np.argpartition(distances, self.k)[:self.k]
        nearest_labels = self.y_train[nearest_indices]      
        index, counts = np.unique(nearest_labels, return_counts=True)
        return index[np.argmax(counts)]
