import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, n_Epochs=10000, learning_rate=0.001, k=1, error_threshold=1e-6, alpha=0.01, regularization=None):
        self.n_Epochs = n_Epochs
        self.learning_rate = learning_rate
        self.k = k
        self.mse_list = []
        self.variance_list = []
        self.std_list = []
        self.epock_to_converge = n_Epochs
        self.error_threshold = error_threshold
        self.alpha = alpha  
        self.regularization = regularization

    def fit(self, x_train, y_train, create_gif=False):
        self.y_train = y_train
        self.y_train = self.y_train[np.argsort(x_train)]
        x_train = np.sort(x_train)
        self.x_train = x_train.reshape(-1, 1)
        if create_gif:
            self.x_train_gif = self.x_train.copy()

        x_train = x_train.reshape(-1, 1)
        for i in range(2, self.k + 1):
            temp = x_train ** i
            self.x_train = np.hstack((self.x_train, temp))

        self.n_samples = x_train.shape[0]
        self.n_features = self.k
        self.weights = np.zeros(self.n_features)
        self.bias = 0
        for i in range(self.n_Epochs):
            self.update_weights()

            if i > 2:
                if abs(self.mse_list[i - 1] - self.mse_list[i - 2]) < self.error_threshold:
                    self.epock_to_converge = i
                    break

            if create_gif and i % 100 == 0:
                self.save_image(f'./figures/gif_images/{i // 100}.png')

    def update_weights(self):
        y_pred = self.predict(self.x_train)
        mse = np.mean((self.y_train - y_pred) ** 2)
        std = np.std(y_pred)
        variance = np.var(y_pred)
        self.mse_list.append(mse)
        self.variance_list.append(variance)
        self.std_list.append(std)

        dW = (1 / self.n_samples) * np.dot(self.x_train.T, (y_pred - self.y_train))
        db = (1 / self.n_samples) * np.sum(y_pred - self.y_train)

        if self.regularization == 'l1':
            reg_term = self.alpha * np.sign(self.weights)
            self.weights -= self.learning_rate * (dW + reg_term)
        elif self.regularization == 'l2':
            reg_term = self.alpha * self.weights
            self.weights -= self.learning_rate * (dW + reg_term)
        else:
            self.weights -= self.learning_rate * dW

        self.bias -= self.learning_rate * db

    def predict(self, x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            xc = x.copy()
            for i in range(2, self.k + 1):
                temp = x ** i
                xc = np.hstack((xc, temp))
            x = xc
        return np.dot(x, self.weights) + self.bias

    def get_line(self, x):
        w = self.weights
        b = self.bias
        n = len(w)
        y = b
        for i in range(n):
            y += w[i] * x ** (i + 1)
        return y

    def save_image(self, name):
        y = self.y_train
        x = self.x_train_gif
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].scatter(x, y)
        axs[0, 0].plot(x, self.get_line(x), color='red')
        axs[0, 0].set_xlabel('X')
        axs[0, 0].set_ylabel('Y')
        axs[0, 0].set_title('Regression line')
        axs[0, 1].plot(self.mse_list)
        axs[0, 1].set_xlabel('Epochs')
        axs[0, 1].set_ylabel('MSE')
        axs[0, 1].set_title('Epochs vs MSE')
        axs[1, 0].plot(self.variance_list)
        axs[1, 0].set_xlabel('Epochs')
        axs[1, 0].set_ylabel('Variance')
        axs[1, 0].set_title('Epochs vs Variance')
        axs[1, 1].plot(self.std_list)
        axs[1, 1].set_xlabel('Epochs')
        axs[1, 1].set_ylabel('Standard Deviation')
        axs[1, 1].set_title('Epochs vs Standard Deviation')
        plt.savefig(name)
        plt.close()
