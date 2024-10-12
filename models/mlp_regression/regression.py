import numpy as np
import wandb

class MLPR:
    def __init__(self, learning_rate=0.01, n_epochs=200, batch_size=32, neurons_per_layer=None,
                 activation_function='sigmoid', loss_function='mean_squared_error', optimizer='sgd', early_stopping=False, patience=10, gradient_check=False):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.neurons_per_layer = neurons_per_layer
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.patience = patience
        self.gradient_check = gradient_check
        if optimizer == 'mini_batch':
            self.batch_size = batch_size
        elif optimizer == 'sgd':
            self.batch_size = 1
        else:
            self.batch_size = batch_size

    def fit(self, X, Y, X_val=None, Y_val=None):
        self.X = X
        self.Y = Y
        self.n_samples, self.n_features = X.shape
        self.n_outputs = Y.shape[1] if len(Y.shape) > 1 else 1

        self.weights = self.initialize_weights()
        self.biases = self.initialize_biases()

        best_loss = np.inf
        patience_counter = 0

        for epoch in range(self.n_epochs):
            for i in range(0, self.n_samples, self.batch_size):
                X_batch = X[i:i+self.batch_size]
                Y_batch = Y[i:i+self.batch_size]

                self.forward_propagation(X_batch)
                self.backward_propagation(Y_batch)
                self.update_weights()

            if X_val is not None and Y_val is not None:
                Y_val_pred = self.predict(X_val)
                Y_train_pred = self.predict(self.X)
                val_loss = self.compute_loss(Y_val_pred, Y_val)
                train_loss = self.compute_loss(Y_train_pred, self.Y)

                wandb.log({
                    'train_loss': train_loss,
                    'val_loss': val_loss
                })

                if self.early_stopping:
                    if val_loss < best_loss:
                        best_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            if self.gradient_check:
                self.check_gradients()

    def initialize_weights(self):
        weights = {}
        layers = [self.n_features] + self.neurons_per_layer + [self.n_outputs]
        for i in range(len(layers) - 1):
            weights[i] = np.random.randn(layers[i], layers[i + 1]) * 0.1
        return weights

    def initialize_biases(self):
        biases = {}
        layers = [self.n_features] + self.neurons_per_layer + [self.n_outputs]
        for i in range(1, len(layers)):
            biases[i] = np.random.randn(layers[i]) * 0.1
        return biases

    def forward_propagation(self, X):
        self.activations = {}
        self.activations[0] = X
        for i in range(1, len(self.weights) + 1):
            z = np.dot(self.activations[i - 1], self.weights[i - 1]) + self.biases[i]
            if i == len(self.weights):
                self.activations[i] = self.linear(z)
            else:
                self.activations[i] = self.activation(z)

    def backward_propagation(self, Y):
        self.errors = {}
        self.errors[len(self.weights)] = self.activations[len(self.weights)] - Y
        for i in range(len(self.weights) - 1, 0, -1):
            self.errors[i] = np.dot(self.errors[i + 1], self.weights[i].T) * self.activation_derivative(self.activations[i])

    def update_weights(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * np.dot(self.activations[i].T, self.errors[i + 1])
            self.biases[i + 1] -= self.learning_rate * np.sum(self.errors[i + 1], axis=0)

    def compute_loss(self, Y_pred, Y_true):
        return np.mean((Y_true - Y_pred) ** 2)

    def activation(self, x):
        if self.activation_function == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation_function == 'relu':
            return np.maximum(0, x)
        elif self.activation_function == 'tanh':
            return np.tanh(x)

    def activation_derivative(self, x):
        if self.activation_function == 'sigmoid':
            return x * (1 - x)
        elif self.activation_function == 'relu':
            return 1. * (x > 0)
        elif self.activation_function == 'tanh':
            return 1 - x ** 2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def linear(self, x):
        return x

    def predict(self, X):
        self.forward_propagation(X)
        return self.activations[len(self.weights)]
