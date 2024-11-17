import numpy as np
import matplotlib.pyplot as plt

class KDE:
    def __init__(self, kernel='gaussian', bandwidth=1.0):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.data = None

    def kernel_fun(self, distance):
        if self.kernel == 'box':
            return np.where(np.abs(distance) <= 1, 0.5, 0)
        elif self.kernel == 'gaussian':
            return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * distance ** 2)
        elif self.kernel == 'triangular':
            return np.maximum(1 - np.abs(distance), 0)
        
    def fit(self, data):
        self.data = np.asarray(data)

    def predict(self, x):
        distances = np.linalg.norm((self.data - x) / self.bandwidth, axis=1)
        kernel_values = self.kernel_fun(distances)
        return np.mean(kernel_values) / (self.bandwidth ** self.data.shape[1])

    def visualize(self, x_range, y_range, resolution=100):
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        positions = np.c_[X.ravel(), Y.ravel()]
        Z = np.array([])
        for pos in positions:
            Z = np.append(Z, self.predict(pos))
        Z = Z.reshape(X.shape)

        plt.figure(figsize=(8, 8))
        plt.contourf(X, Y, Z, cmap='viridis')
        # plt.scatter(self.data[:, 0], self.data[:, 1], s=5, color='red')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'KDE Density Visualization ({self.kernel} kernel, bandwidth={self.bandwidth})')
        plt.colorbar(label='Density')
        plt.savefig(f'./figures/KDE_{self.kernel}_kernel_{int(self.bandwidth*100)}.png')
