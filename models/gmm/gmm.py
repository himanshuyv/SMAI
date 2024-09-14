import numpy as np
from scipy.stats import multivariate_normal

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
            # print("Iteration: ", _)
            # print(self.pi)
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
