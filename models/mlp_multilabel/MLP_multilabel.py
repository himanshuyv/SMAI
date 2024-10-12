import numpy as np
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder

class MLP_multilabel:
    def __init__(self, learning_rate=0.01, n_epochs=200, batch_size=32, neurons_per_layer=None,
                 activation_function='sigmoid', loss_function='binary_cross_entropy', optimizer='sgd', early_stopping=False, patience=10):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.neurons_per_layer = neurons_per_layer
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.patience = patience
        if optimizer == 'mini-batch':
            self.batch_size = batch_size
        elif optimizer == 'sgd':
            self.batch_size = 1

    def fit(self, X, Y, X_val=None, Y_val=None):
        self.X = X
        self.Y = Y
        self.n_samples, self.n_features = X.shape
        self.n_classes = Y.shape[1]

        if self.optimizer == 'batch':
            self.batch_size = self.n_samples
        self.weights, self.biases = self.initialize_weightsAndBiases()

        best_loss = np.inf
        patience_counter = 0

        for epoch in range(self.n_epochs):
            for i in range(0, self.n_samples, self.batch_size):
                X_batch = X[i:i+self.batch_size]
                Y_batch = Y[i:i+self.batch_size]

                self.forward_propagation(X_batch)
                grads_w, grads_b = self.backward_propagation(Y_batch)
                self.update_weights(grads_w, grads_b)

            # loss = self.compute_loss(self.X, self.Y)
            # print(f"Epoch {epoch+1}/{self.n_epochs} - Loss: {loss}")

            if X_val is not None and Y_val is not None:
                Y_val_pred = self.predict(X_val)
                Y_train_pred = self.predict(self.X)
                val_loss = self.compute_loss(X_val, Y_val)
                train_loss = self.compute_loss(self.X, self.Y)
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

    def initialize_weightsAndBiases(self):
        weights = {}
        biases = {}
        layers = [self.n_features] + self.neurons_per_layer + [self.n_classes]
        for i in range(len(layers) - 1):
            weights[i] = np.random.randn(layers[i], layers[i + 1]) 
            if self.activation_function == 'relu':
                weights[i] *= np.sqrt(2 / layers[i])
            else:
                weights[i] *= np.sqrt(1 / layers[i])
            biases[i] = np.zeros((1, layers[i+1]))
        return weights, biases

    def forward_propagation(self, X):
        self.activations = {}
        self.activations[0] = X
        for i in range(1, len(self.weights) + 1):
            z = np.dot(self.activations[i - 1], self.weights[i - 1]) + self.biases[i - 1]
            if i == len(self.weights):
                self.activations[i] = self.sigmoid(z)
            else:
                self.activations[i] = self.activation(z)

    def sigmoid(self, x):
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
        
    def activation_derivative(self, x):
        if self.activation_function == 'sigmoid':
            return x * (1 - x)
        elif self.activation_function == 'relu':
            return (x > 0).astype(int)
        elif self.activation_function == 'tanh':
            return 1 - x ** 2
        elif self.activation_function == 'linear':
            return np.ones_like(x)

    def backward_propagation(self, Y):
        self.errors = {}
        self.errors[len(self.weights)] = self.activations[len(self.weights)] - Y
        
        M = Y.shape[0]
        grads_w = {}
        grads_b = {}

        for i in range(len(self.weights), 0, -1):
            grads_w[i - 1] = np.dot(self.activations[i - 1].T, self.errors[i]) / M
            grads_b[i - 1] = np.sum(self.errors[i], axis=0, keepdims=True) / M
            if i > 1:
                self.errors[i - 1] = np.dot(self.errors[i], self.weights[i - 1].T) * self.activation_derivative(self.activations[i - 1])
        return grads_w, grads_b

    def update_weights(self, grads_w, grads_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    def compute_loss(self, X, Y):
        self.forward_propagation(X)
        Y_pred = self.activations[len(self.weights)]
        Y_pred = np.clip(Y_pred, 1e-10, 1 - 1e-10)
        M = Y.shape[0]
        if self.loss_function == 'binary_cross_entropy':
            return -np.sum(Y * np.log(Y_pred + 1e-10) + (1 - Y) * np.log(1 - Y_pred + 1e-10)) / M
        elif self.loss_function == 'mean_squared_error':
            return np.mean((Y - Y_pred) ** 2)

    def compute_metrics(self, Y_pred, Y_true):
        Y_pred_binary = (Y_pred >= 0.5).astype(int)
        metrics = {
            'accuracy': accuracy_score(Y_true, Y_pred_binary),
            'precision': precision_score(Y_true, Y_pred_binary, average='macro', zero_division=0),
            'recall': recall_score(Y_true, Y_pred_binary, average='macro', zero_division=0),
            'f1': f1_score(Y_true, Y_pred_binary, average='macro', zero_division=0)
        }
        return metrics

    def predict(self, X):
        self.forward_propagation(X)
        output = self.activations[len(self.weights)]
        return (output >= 0.5).astype(int)