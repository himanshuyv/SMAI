import numpy as np

class LinearRegression:
    def __init__(self, n_iterations=100, learning_rate=0.001, k = 1 ,lambda_= 0.0):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.k = k
    
    def fit(self, x_train, y_train):
        self.x_train = x_train.reshape(-1,1)
        x_train = x_train.reshape(-1,1)
        for i in range(2,self.k+1):
            temp = x_train ** i
            self.x_train = np.hstack((self.x_train,temp))

        self.y_train = y_train
        self.n_samples = x_train.shape[0]
        self.n_features = self.k
        self.weights = np.zeros(self.n_features)
        self.bias = 0
        for i in range(self.n_iterations):
            self.update_weights()

    def update_weights(self):
        y_pred = self.predict(self.x_train)
        dW = (1/self.n_samples) * np.dot(self.x_train.T , (y_pred - self.y_train))
        self.weights -= self.learning_rate * dW
        db = (1/self.n_samples) * np.sum(y_pred - self.y_train)
        self.bias -= self.learning_rate * db

    def predict(self, x):
        if x.ndim == 1:
            x = x.reshape(-1,1)
            xc = x.copy()
            for i in range(2,self.k+1):
                temp = x ** i
                xc = np.hstack((xc,temp))
            x = xc
        return np.dot(x,self.weights) + self.bias