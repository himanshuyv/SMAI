# Assignment 2

## K-means clustering

### Task 1: K-means class
```python
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
```

- Clustering Plot for the smaller dataset
    ![Clustering Plot](./figures/kmeans_small.png)

### Task 2: Elbow Method

- Elbow method for the smaller dataset
![Elbow Method](./figures/kmeans_elbow_method_small.png)

Here from plot we can see that the elbow point is at k = 3.

- Elbow method for the larger dataset
![Elbow Method](./figures/kmeans_elbow_method_big.png)

It depends on the different runs, as in the class I am initializing the centroids randomly. For this plot the elbow point is at k = 11.

Hence ```k_kmeans1 = 11```

On performing the K-means clustering on the larger dataset with k = 11, the Within-Cluster Sum of Squares (WCSS) cost is 3723.25035998532.

But this cost also varies for different runs, as the centroids are initialized randomly.

## Gaussian Mixture Model

### Task 1: GMM class
```python
class Gmm:
    def __init__(self, k=3, n_iter=100):
        self.k = k
        self.n_iter = n_iter
        self.prev_likelihood = 0

    def fit(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        self.pi = np.ones(self.k) / self.k
        self.mu = X[np.random.choice(self.n_samples, self.k, replace=False)]
        self.sigma = np.array([np.eye(self.n_features)] * self.k)

        for _ in range(self.n_iter):
            self.e_step()
            self.m_step()
            likelihood = self.getLikelihood()
            if abs(likelihood - self.prev_likelihood) < 1e-6:
                break
            self.prev_likelihood = likelihood

    def e_step(self):
        self.gamma = np.zeros((self.n_samples, self.k))
        log_probs = np.zeros((self.n_samples, self.k))
        
        for j in range(self.k):
            mvn = multivariate_normal(mean=self.mu[j], cov=self.sigma[j])
            log_probs[:, j] = np.log(self.pi[j]) + mvn.logpdf(self.X)
        
        log_probs_max = np.max(log_probs, axis=1, keepdims=True)
        log_probs -= log_probs_max 
        probs = np.exp(log_probs)
        self.gamma = probs / np.sum(probs, axis=1, keepdims=True)

    def m_step(self):
        N = np.sum(self.gamma, axis=0)
        N = np.where(N < 1e-6, 1e-6, N)
        self.pi = N / self.n_samples
        self.mu = np.dot(self.gamma.T, self.X) / N[:, None]
        
        for j in range(self.k):
            x_mu = self.X - self.mu[j]
            weighted_cov = np.dot(self.gamma[:, j] * x_mu.T, x_mu) / N[j]
            self.sigma[j] = weighted_cov + np.eye(self.n_features) * 1e-6

    def get_params(self):
        return self.pi, self.mu, self.sigma
    
    def getMembership(self):
        return self.gamma
    
    def getLikelihood(self):
        log_likelihood = 0
        log_probs = np.zeros((self.n_samples, self.k))
        
        for j in range(self.k):
            mvn = multivariate_normal(mean=self.mu[j], cov=self.sigma[j])
            log_probs[:, j] = np.log(self.pi[j]) + mvn.logpdf(self.X)
        
        log_probs_max = np.max(log_probs, axis=1, keepdims=True)
        log_probs -= log_probs_max
        probs = np.exp(log_probs)
        log_likelihood = np.sum(log_probs_max.squeeze() + np.log(np.sum(probs, axis=1)))
        
        return log_likelihood

    def get_num_params(self):
        num_means = self.k * self.n_features
        num_covariances = self.k * (self.n_features * (self.n_features + 1)) // 2 
        num_weights = self.k - 1
        return num_means + num_covariances + num_weights

    def aic(self):
        num_params = self.get_num_params()
        log_likelihood = self.getLikelihood()
        return 2 * num_params - 2 * log_likelihood
    
    def bic(self):
        num_params = self.get_num_params()
        log_likelihood = self.getLikelihood()
        return num_params * np.log(self.n_samples) - 2 * log_likelihood
```

Here Initially I implemented the GMM class with the formula given in class because of the overflow I took help from LLM to implement the log probabilities. 

So the Current version of the E-step and the M-step is taken from the LLM.

### Task 2: Testing, AIC and BIC

```
Likelihood:  522241.37061497767
pi:  [0.245 0.375 0.38 ]
Inbuilt GMM Likelihood:  575859.1572580193
```
As we can see from the above results, the likelihood of my gmm is less than the inbuilt gmm (Although they are close). This is because we initialize the mu and sigma randomly and the likelihood depends on the initialization of the mu and sigma. The above results are for the best k that is 3.

- Plot for the smaller dataset.
![GMM Plot](./figures/gmm_clusters.png)

- AIC and BIC for the larger dataset
![AIC and BIC](./figures/gmm_aic_bic_mine.png)

- AIC and BIC for the larger dataset using inbuilt function
![AIC and BIC](./figures/gmm_aic_bic_inbuilt.png)

From the above plots, Elbow point is not very clear for this dataset. But most closer to the elbow point is at k = 2 of k = 3.

Hence ```k_gmm1 = 3```


## Dimensionality Reduction and Visualization

### Task 1: PCA class
```python
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
```

###  Perform Dimensionality Reduction

- keeping the `error_threshold = 0.1` for both 2D and 3D checkPCA returns True.
```
Before Transformation: True
(200, 512)
After PCA
(200, 2)
Before Transformation: True
(200, 512)
After PCA
(200, 3)
``` 

- Plot for 2D
![PCA Plot](./figures/pca_2d.png)

- Plot for 3D
![PCA Plot](./figures/pca_3d.png)

### Data Analysis

- The new axes represent the directions of maximum variance in the data.

- When we look at the 2D plot, then we can identify 2 or 3 clusters. But when we look at the 3D plot, then we can identify 4 clusters. Hence taking `k2 = 4`.


## PCA + Clustering
### K-means Clustering Based on 2D Visualization
- K-means clustering on the 2D PCA transformed data with k = k2, and number of dimensions = 2.

![PCA Plot](./figures/pca_kmeans.png)

### PCA + K-Means Clustering
- Scree Plot
![PCA Plot](./figures/pca_explained_variance.png)

- Eigen Values Plot
![PCA Plot](./figures/pca_eigenvalues_fraction.png)

To capture 95% of the variance, we need 132 dimensions.

- Elbow Method
![Elbow Method](./figures/pca_kmeans_elbow.png)

From the above plot, the elbow point is at k = 8.
Hence `k_kmeans3 = 8`.

- The cost(WCSS) for k = k_kmeans3 for reduced Dataset: 3633.909127962344

### GMM Clustering Based on 2D Visualization

- GMM clustering on the 2D PCA transformed data with k = k2, and number of dimensions = 2.

![PCA Plot](./figures/pca_gmm_clusters.png)

###  PCA + GMMs

- AIC and BIC for the reduced dataset
![AIC and BIC](./figures/pca_gmm_aic_bic.png)

From the above plots, the elbow point is at k = 4. It vary for different runs but most of the time it is at k = 4 of k = 5.
Hence `k_gmm3 = 4`.

- With k = k_gmm3, the Likelihood for reduced Dataset:  89705.35369496288.

## Cluster Analysis

###  K- Means Cluster Analysis


## Hierarchical Clustering

Plots For different linkage methods and different distance metrics

- Single Linkage with Euclidean Distance
![Single Linkage](./figures/dendrogram_single_euclidean.png)

- Single Linkage with Cosine Distance
![Single Linkage](./figures/dendrogram_single_cosine.png)

- Complete Linkage with Euclidean Distance
![Complete Linkage](./figures/dendrogram_complete_euclidean.png)

- Complete Linkage with Cosine Distance
![Complete Linkage](./figures/dendrogram_complete_cosine.png)

- Average Linkage with Euclidean Distance
![Average Linkage](./figures/dendrogram_average_euclidean.png)

- Average Linkage with Cosine Distance
![Average Linkage](./figures/dendrogram_average_cosine.png)

- Ward Linkage with Euclidean Distance
![Ward Linkage](./figures/dendrogram_ward_euclidean.png)

- Centroid Linkage with Euclidean Distance
![Centroid Linkage](./figures/dendrogram_centroid_euclidean.png)


## Nearst Neighbour Search

### Task 1: PCA + KNN

Scree plot for the Spotify dataset

- Cumilative Explained Variance
![PCA Plot](./figures/pca_knn_explained_variance.png)

- Eigen Values Plot
![PCA Plot](./figures/pca_knn_eigenvalues_fraction.png)

To capture 95% of the variance, we need 8 features.


### Evaluation
- Keeping all the features, with best k = 20 and best distance metric = 'manhattan', the scores are as follows:
```
Full Dataset
Accuracy: 0.2709343138893532
Micro Precision: 0.2709343138893532
Micro Recall: 0.2709343138893532
Micro F1: 0.2709343138893532
Macro Precision: 0.2573578888671874
Macro Recall: 0.2478699674397185
Macro F1: 0.24091608277387128
```
- Keeping the best 8 features, with best k = 20 and best distance metric = 'manhattan', the scores are as follows:
```
Reduced Dataset
Accuracy: 0.21232380634018608
Micro Precision: 0.21232380634018608
Micro Recall: 0.21232380634018608
Micro F1: 0.21232380634018608
Macro Precision: 0.2023337961950314
Macro Recall: 0.19550068347367766
Macro F1: 0.1896011645793315
```

We can see that the accuracy is less for the reduced dataset as compared to the full dataset. This is because we are reducing the dimensions and hence losing some information. But using only 8 features we are able to capute 95% of the variance and got a good accuracy.

Same is the case with the precision, recall and F1 scores.

Reducing the dimensions also reduces the inference time.

![Inference Time](./figures/pca_knn_inference_time.png)

As we can see from the bar plot that the inference time for the reuced dataset is less as compared to the full dataset.