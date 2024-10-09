import numpy as np

class MLP:
    def __init__(self, learning_rate=0.01, n_epochs=1000, n_hidden=2, batch_size=32, neurons_per_layer=None,
                 activation_function='sigmoid', loss_function='mean_squared_error', optimizer='sgd', early_stopping=False):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.neurons_per_layer = neurons_per_layer if neurons_per_layer else [10] * n_hidden
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.best_weights = None
        self.best_biases = None
        self.best_loss = float('inf')
        self.patience = 10


    def fit(self, X, Y, X_val=None, Y_val=None):
        self.X = X
        self.Y = Y
        self.n_samples, self.n_features = X.shape
        self.n_classes = np.max(Y) + 1

        self.weights = self.initialize_weights()
        self.biases = self.initialize_biases()

        stop_counter = 0
        for epoch in range(self.n_epochs):
            for i in range(0, self.n_samples, self.batch_size):
                X_batch = X[i:i+self.batch_size]
                Y_batch = Y[i:i+self.batch_size]
                Y_batch_one_hot = self.one_hot_encode(Y_batch) if self.n_classes > 1 else Y_batch.reshape(-1, 1)
                
                self.forward_propagation(X_batch)
                self.backward_propagation(Y_batch_one_hot)
                self.update_weights()

            if self.early_stopping and X_val is not None and Y_val is not None:
                val_loss = self.compute_loss(self.predict(X_val), Y_val)
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.best_weights = {k: v.copy() for k, v in self.weights.items()}
                    self.best_biases = {k: v.copy() for k, v in self.biases.items()}
                    stop_counter = 0
                else:
                    stop_counter += 1
                    if stop_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch}")
                        self.weights = self.best_weights
                        self.biases = self.best_biases
                        break

    def initialize_weights(self):
        weights = {}
        layers = [self.n_features] + self.neurons_per_layer + [self.n_classes]
        for i in range(len(layers) - 1):
            weights[i] = np.random.randn(layers[i], layers[i+1]) * 0.1
        return weights

    def initialize_biases(self):
        biases = {}
        layers = [self.n_features] + self.neurons_per_layer + [self.n_classes]
        for i in range(1, len(layers)):
            biases[i] = np.random.randn(layers[i]) * 0.1
        return biases

    def forward_propagation(self, X):
        self.activations = {}
        self.activations[0] = X
        for i in range(1, len(self.weights) + 1):
            z = np.dot(self.activations[i-1], self.weights[i-1]) + self.biases[i]
            self.activations[i] = self.activation(z)

    def activation(self, x):
        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == 'relu':
            return np.maximum(0, x)
        elif self.activation_function == 'tanh':
            return np.tanh(x)
        elif self.activation_function == 'linear':
            return x

    def activation_derivative(self, x):
        if self.activation_function == 'sigmoid':
            return x * (1 - x)
        elif self.activation_function == 'relu':
            return 1. * (x > 0)
        elif self.activation_function == 'tanh':
            return 1 - x ** 2
        elif self.activation_function == 'linear':
            return 1

    def backward_propagation(self, Y):
        self.errors = {}
        self.errors[len(self.weights)] = Y - self.activations[len(self.weights)]
        for i in range(len(self.weights) - 1, 0, -1):
            self.errors[i] = np.dot(self.errors[i + 1], self.weights[i].T) * self.activation_derivative(self.activations[i])

    def update_weights(self):
        for i in range(len(self.weights)):
            if self.optimizer == 'sgd':
                self.weights[i] += self.learning_rate * np.dot(self.activations[i].T, self.errors[i + 1])
                self.biases[i + 1] += self.learning_rate * np.sum(self.errors[i + 1], axis=0)

    def compute_loss(self, Y_pred, Y_true):
        if self.loss_function == 'mean_squared_error':
            if self.n_classes > 1:
                Y_true_one_hot = self.one_hot_encode(Y_true)
                return np.mean((Y_pred - Y_true_one_hot) ** 2)
            else:
                return np.mean((Y_pred - Y_true.reshape(-1, 1)) ** 2)

    def predict(self, X):
        self.forward_propagation(X)
        output = self.activations[len(self.weights)]
        if self.n_classes > 1:
            return np.argmax(output, axis=1)
        
        elif self.n_classes == 1:
            return (output >= 0.5).astype(int)

    def one_hot_encode(self, Y):
        one_hot = np.zeros((Y.size, self.n_classes))
        one_hot[np.arange(Y.size), Y] = 1
        return one_hot

    