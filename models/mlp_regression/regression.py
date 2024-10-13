import numpy as np
import wandb

class MLPR:
    def __init__(self, learning_rate=0.01, n_epochs=200, batch_size=32, neurons_per_layer=None,
                 activation_function='sigmoid', loss_function='mean_squared_error', optimizer='sgd', early_stopping=False, patience=30):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.neurons_per_layer = neurons_per_layer
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.patience = patience

    def fit(self, X, Y, X_val=None, Y_val=None):
        self.X = X
        self.Y = Y
        self.n_samples, self.n_features = X.shape
        self.n_classes = Y.shape[1] if len(Y.shape) > 1 else 1

        if self.optimizer == 'batch':
            self.batch_size = self.n_samples

        self.weights, self.biases = self.initialize_weights_and_biases()

        best_loss = np.inf
        patience_counter = 0
        self.loss_list = []
        for epoch in range(self.n_epochs):
            for i in range(0, self.n_samples, self.batch_size):
                X_batch = X[i:i + self.batch_size]
                Y_batch = Y[i:i + self.batch_size]

                # print('batch_size:',self.batch_size)
                # print('X_batch_shape:',X_batch.shape)
                # print('Y_batch_shape:',Y_batch.shape)
                self.forward_propagation(X_batch)
                grads_w, grads_b = self.backward_propagation(Y_batch)
                self.update_weights(grads_w, grads_b)

            loss = self.compute_loss(self.X, self.Y)
            self.loss_list.append(loss)
            # print(f"Epoch {epoch + 1}/{self.n_epochs} - Loss: {loss}")

            if X_val is not None and Y_val is not None:
                val_loss = self.compute_loss(X_val, Y_val.reshape(-1, 1))
                train_loss = self.compute_loss(self.X, self.Y)
                Y_val_pred = self.predict(X_val)
                Y_train_pred = self.predict(self.X)
                metrics_train = self.compute_metrics(Y_train_pred, self.Y)
                metrics_val = self.compute_metrics(Y_val_pred, Y_val.reshape(-1, 1))

                wandb.log({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_rmse': metrics_train['rmse'],
                    'val_rmse': metrics_val['rmse'],
                    'train_mae': metrics_train['mae'],
                    'val_mae': metrics_val['mae'],
                    'train_r_squared': metrics_train['r_squared'],
                    'val_r_squared': metrics_val['r_squared']
                })

                if self.early_stopping:
                    if val_loss < best_loss:
                        best_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

    def initialize_weights_and_biases(self):
        weights = {}
        biases = {}
        layers = [self.n_features] + self.neurons_per_layer + [self.n_classes]
        for i in range(len(layers) - 1):
            weights[i] = np.random.randn(layers[i], layers[i + 1]) * 0.1
            biases[i] = np.zeros((1, layers[i + 1]))
        return weights, biases

    def forward_propagation(self, X):
        self.activations = {}
        self.activations[0] = X
        self.z = {}
        for i in range(1, len(self.weights) + 1):
            z = np.dot(self.activations[i - 1], self.weights[i - 1]) + self.biases[i - 1]
            self.activations[i] = self.activation(z)
            self.z[i - 1] = z

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def activation(self, x):
        if self.activation_function == 'sigmoid' or self.loss_function == 'binary_cross_entropy':
            return self.sigmoid(x)
        elif self.activation_function == 'relu':
            return np.maximum(0, x)
        elif self.activation_function == 'tanh':
            return np.tanh(x)
        elif self.activation_function == 'linear':
            return x

    def activation_derivative(self, x):
        if self.activation_function == 'sigmoid' or self.loss_function == 'binary_cross_entropy':
            return self.sigmoid(x) * (1 - self.sigmoid(x))
        elif self.activation_function == 'relu':
            return 1. * np.where(x > 0, 1, 0)
        elif self.activation_function == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_function == 'linear':
            return 1

    def backward_propagation(self, Y):
        error = self.activations[len(self.weights)] - Y
        
        M = Y.shape[0]

        grads_w = {}
        grads_b = {}

        for i in range(len(self.weights)-1, -1, -1):
            grads_w[i] = np.dot(self.activations[i].T, error)/M  
            grads_b[i] = np.sum(error, axis=0, keepdims=True)/M
            if i > 0:
                error = np.dot(error, self.weights[i].T) * self.activation_derivative(self.z[i-1])
        return grads_w, grads_b

    def update_weights(self, grads_w, grads_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    def compute_loss(self, X, Y):
        self.forward_propagation(X)
        Y_pred = self.activations[len(self.weights)]
        if self.loss_function == 'mean_squared_error':
            return np.mean((Y - Y_pred) ** 2)
        elif self.loss_function == 'mean_absolute_error':
            return np.mean(np.abs(Y - Y_pred))
        elif self.loss_function == 'binary_cross_entropy':
            epsilon = 1e-12
            Y_pred = np.clip(Y_pred, epsilon, 1. - epsilon)
            return -np.mean(Y * np.log(Y_pred) + (1 - Y) * np.log(1 - Y_pred))
        
    def mean_squared_error(self, Y_true, Y_pred):
        return np.mean((Y_true - Y_pred) ** 2)
    
    def mean_absolute_error(self, Y_true, Y_pred):
        return np.mean(np.abs(Y_true - Y_pred))
    
    def r_squared(self, Y_true, Y_pred):
        ss_total = np.sum((Y_true - np.mean(Y_true)) ** 2)
        ss_residual = np.sum((Y_true - Y_pred) ** 2)
        if ss_total == 0:
            return 0
        return 1 - (ss_residual / ss_total)

    def compute_metrics(self, Y_pred, Y_true):
        metrics = {
            'mse': self.mean_squared_error(Y_true, Y_pred),
            'rmse': np.sqrt(self.mean_squared_error(Y_true, Y_pred)),
            'mae': self.mean_absolute_error(Y_true, Y_pred),
            'r_squared': self.r_squared(Y_true, Y_pred)
        }
        return metrics

    def predict(self, X):
        self.forward_propagation(X)
        return self.activations[len(self.weights)]
    
    def gradient_check(self, X, Y, epsilon=1e-7):
        self.forward_propagation(X)
        grads_w, grads_b = self.backward_propagation(Y)

        numerical_grads_w = {}
        numerical_grads_b = {}

        for i in range(len(self.weights)):
            numerical_grads_w[i] = np.zeros(self.weights[i].shape)
            for j in range(self.weights[i].shape[0]):
                for k in range(self.weights[i].shape[1]):
                    self.weights[i][j, k] += epsilon
                    loss_plus = self.compute_loss(X, Y)

                    self.weights[i][j, k] -= 2 * epsilon
                    loss_minus = self.compute_loss(X, Y)

                    self.weights[i][j, k] += epsilon
                    numerical_grads_w[i][j, k] = (loss_plus - loss_minus) / (2 * epsilon)

        for i in range(len(self.biases)):
            numerical_grads_b[i] = np.zeros(self.biases[i].shape)
            for j in range(self.biases[i].shape[1]):
                self.biases[i][0, j] += epsilon
                loss_plus = self.compute_loss(X, Y)

                self.biases[i][0, j] -= 2 * epsilon
                loss_minus = self.compute_loss(X, Y)

                self.biases[i][0, j] += epsilon
                numerical_grads_b[i][0, j] = (loss_plus - loss_minus) / (2 * epsilon)

        for i in range(len(self.weights)):
            diff_w = np.linalg.norm(grads_w[i] - numerical_grads_w[i]) / (np.linalg.norm(grads_w[i]) + np.linalg.norm(numerical_grads_w[i]) + 1e-8)
            if diff_w > 1e-4:
                print(f"Gradient check failed for weights at layer {i} with difference {diff_w}")
                return

        for i in range(len(self.biases)):
            diff_b = np.linalg.norm(grads_b[i] - numerical_grads_b[i]) / (np.linalg.norm(grads_b[i]) + np.linalg.norm(numerical_grads_b[i]) + 1e-8)
            if diff_b > 1e-4:
                print(f"Gradient check failed for biases at layer {i} with difference {diff_b}")
                return

        print("Gradient check passed")
