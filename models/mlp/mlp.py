import numpy as np
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder

class MLP:
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
        if (optimizer == 'mini_batch'):
            self.batch_size = batch_size
        elif (optimizer == 'sgd'):
            self.batch_size = 1
        else:
            self.batch_size = batch_size
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)

    def fit(self, X, Y, X_val=None, Y_val=None):
        self.X = X
        self.Y = Y
        self.n_samples, self.n_features = X.shape
        self.n_classes = len(np.unique(Y)) if len(np.unique(Y)) > 2 else 1

        self.weights = self.initialize_weights()
        self.biases = self.initialize_biases()

        if self.n_classes > 1:
            Y = self.one_hot_encoder.fit_transform(Y.reshape(-1, 1))

        best_loss = np.inf
        patience_counter = 0

        for epoch in range(self.n_epochs):
            for i in range(0, self.n_samples, self.batch_size):
                X_batch = X[i:i+self.batch_size]
                Y_batch = Y[i:i+self.batch_size]

                if self.n_classes > 1:
                    Y_batch_one_hot = Y_batch
                else:
                    Y_batch_one_hot = Y_batch.reshape(-1, 1)

                self.forward_propagation(X_batch)
                self.backward_propagation(Y_batch_one_hot)
                self.update_weights()

            if X_val is not None and Y_val is not None:
                Y_val_pred = self.predict(X_val)
                Y_train_pred = self.predict(self.X)
                val_loss = self.compute_loss(Y_val_pred, Y_val)
                train_loss = self.compute_loss(Y_train_pred, self.Y)
                score_train = self.compute_metrics(Y_train_pred, self.Y)
                score_val = self.compute_metrics(Y_val_pred, Y_val)

                wandb.log({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_accuracy': score_train['accuracy'],
                    'val_accuracy': score_val['accuracy'],
                    'train_precision': score_train['precision'],
                    'val_precision': score_val['precision'],
                    'train_recall': score_train['recall'],
                    'val_recall': score_val['recall'],
                    'train_f1': score_train['f1'],
                    'val_f1': score_val['f1']
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
        layers = [self.n_features] + self.neurons_per_layer + [self.n_classes]
        for i in range(len(layers) - 1):
            weights[i] = np.random.randn(layers[i], layers[i + 1]) * 0.1
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
            z = np.dot(self.activations[i - 1], self.weights[i - 1]) + self.biases[i]
            if i == len(self.weights): 
                if self.n_classes == 1:
                    self.activations[i] = self.sigmoid(z)
                else:
                    self.activations[i] = self.softmax(z)
            else:
                self.activations[i] = self.activation(z)

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def activation(self, x):
        if self.activation_function == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation_function == 'relu':
            return np.maximum(0, x)
        elif self.activation_function == 'tanh':
            return np.tanh(x)
        elif self.activation_function == 'linear':
            return x

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def activation_derivative(self, x):
        if self.activation_function == 'sigmoid':
            z = self.sigmoid(x)
            return z * (1 - z)
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
            self.weights[i] += self.learning_rate * np.dot(self.activations[i].T, self.errors[i + 1])
            self.biases[i + 1] += self.learning_rate * np.sum(self.errors[i + 1], axis=0)

    def compute_loss(self, Y_pred, Y_true):
        if self.n_classes > 1 and len(Y_true.shape) == 1:
            Y_true = self.one_hot_encoder.transform(Y_true.reshape(-1, 1))

        if self.loss_function == 'cross_entropy':
            if self.n_classes > 1:
                return -np.mean(np.sum(Y_true * np.log(Y_pred + 1e-8), axis=1))
            else:
                Y_true = Y_true.reshape(-1, 1)
                return -np.mean(Y_true * np.log(Y_pred + 1e-8) + (1 - Y_true) * np.log(1 - Y_pred + 1e-8))

        elif self.loss_function == 'mean_squared_error':
            return np.mean((Y_true - Y_pred) ** 2)

    def compute_metrics(self, Y_pred, Y_true):
        if self.n_classes > 1:
            if len(Y_true.shape) > 1 and Y_true.shape[1] > 1:
                Y_true = np.argmax(Y_true, axis=1)
            if len(Y_pred.shape) > 1 and Y_pred.shape[1] > 1:
                Y_pred = np.argmax(Y_pred, axis=1)
        metrics = {
            'accuracy': accuracy_score(Y_true, Y_pred),
            'precision': precision_score(Y_true, Y_pred, average='macro', zero_division=0),
            'recall': recall_score(Y_true, Y_pred, average='macro', zero_division=0),
            'f1': f1_score(Y_true, Y_pred, average='macro', zero_division=0)
        }
        return metrics

    def predict(self, X):
        self.forward_propagation(X)
        output = self.activations[len(self.weights)]
        if self.n_classes > 1:
            predicted_classes = np.argmax(output, axis=1)
            return self.one_hot_encoder.inverse_transform(np.eye(self.n_classes)[predicted_classes].reshape(-1, self.n_classes))
        elif self.n_classes == 1:
            return (output >= 0.5).astype(int)

    def check_gradients(self, epsilon=1e-7):
        print("Performing gradient checking...")
        for i in range(len(self.weights)):
            for w_idx in np.ndindex(self.weights[i].shape):
                original_weight = self.weights[i][w_idx]
                self.weights[i][w_idx] += epsilon
                plus_loss = self.compute_loss(self.predict(self.X), self.Y)
                self.weights[i][w_idx] -= 2 * epsilon
                minus_loss = self.compute_loss(self.predict(self.X), self.Y)
                self.weights[i][w_idx] = original_weight

                numerical_gradient = (plus_loss - minus_loss) / (2 * epsilon)
                backprop_gradient = np.dot(self.activations[i].T, self.errors[i + 1])[w_idx]

                if not np.isclose(numerical_gradient, backprop_gradient, atol=1e-5):
                    print(f"Gradient check failed at layer {i}, weight {w_idx}.")
                    print(f"Numerical: {numerical_gradient}, Backprop: {backprop_gradient}")
                    return False
        print("Gradient check passed!")
        return True
