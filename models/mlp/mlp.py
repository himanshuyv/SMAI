import numpy as np
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder

class MLP:
    def __init__(self, learning_rate=0.01, n_epochs=200, batch_size=32, neurons_per_layer=None,
                 activation_function='sigmoid', loss_function='mean_squared_error', optimizer='sgd', early_stopping=False, patience=10):
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
            self.batch_size = 1``

        self.one_hot_encoder = OneHotEncoder(sparse_output=False)

    def fit(self, X, Y, X_val=None, Y_val=None):
        self.X = X
        self.Y = Y
        self.n_samples, self.n_features = X.shape
        self.n_classes = len(np.unique(Y)) if len(np.unique(Y)) > 2 else 1

        if self.optimizer == 'batch':
            self.batch_size = self.n_samples
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
                grads_w, grads_b = self.backward_propagation(Y_batch_one_hot)
                self.update_weights(grads_w, grads_b)

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
            return x * (1 - x)
        elif self.activation_function == 'relu':
            return 1. * (x > 0)
        elif self.activation_function == 'tanh':
            return 1 - x ** 2
        elif self.activation_function == 'linear':
            return 1

    def backward_propagation(self, Y):
        self.errors = {}
        self.errors[len(self.weights)] = self.activations[len(self.weights)] - Y
        
        grads_w = {}
        grads_b = {}

        for i in range(len(self.weights), 0, -1):
            grads_w[i - 1] = np.dot(self.activations[i - 1].T, self.errors[i])  
            grads_b[i] = np.sum(self.errors[i], axis=0, keepdims=True)
            if i > 1:
                self.errors[i - 1] = np.dot(self.errors[i], self.weights[i - 1].T) * self.activation_derivative(self.activations[i - 1])
        return grads_w, grads_b


    def update_weights(self, grads_w, grads_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i + 1] -= self.learning_rate * grads_b[i + 1].reshape(-1)


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

    def predict_proba(self, X):
        self.forward_propagation(X)
        return self.activations[len(self.weights)]

    def gradient_check(self, X, Y, epsilon=1e-7):
        self.forward_propagation(X)
        Y = self.one_hot_encoder.transform(Y.reshape(-1, 1))
        grads_w, grads_b = self.backward_propagation(Y)
        
        numerical_grads_w = {}
        numerical_grads_b = {}

        for l in range(len(self.weights)):
            numerical_grads_w[l] = np.zeros_like(self.weights[l])
            for i in range(self.weights[l].shape[0]):
                for j in range(self.weights[l].shape[1]):
                    original_weight = self.weights[l][i, j]
                    self.weights[l][i, j] = original_weight + epsilon
                    self.forward_propagation(X)
                    loss_plus_epsilon = self.compute_loss(self.activations[len(self.weights)], Y)
                    self.weights[l][i, j] = original_weight - epsilon
                    self.forward_propagation(X)
                    loss_minus_epsilon = self.compute_loss(self.activations[len(self.weights)], Y)
                    self.weights[l][i, j] = original_weight
                    numerical_grads_w[l][i, j] = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)

            relative_difference_w = np.linalg.norm(grads_w[l] - numerical_grads_w[l]) / (
                np.linalg.norm(grads_w[l]) + np.linalg.norm(numerical_grads_w[l]) + 1e-8
            )
            if relative_difference_w > 1e-6:
                print(f"Gradient check failed for weights in layer {l + 1}: relative difference is {relative_difference_w}")
                return False

        for l in range(1, len(self.biases) + 1):
            numerical_grads_b[l] = np.zeros_like(self.biases[l])
            
            for i in range(len(self.biases[l])):
                original_bias = self.biases[l][i]
                self.biases[l][i] = original_bias + epsilon
                self.forward_propagation(X)
                loss_plus_epsilon = self.compute_loss(self.activations[len(self.weights)], Y)
                
                self.biases[l][i] = original_bias - epsilon
                self.forward_propagation(X)
                loss_minus_epsilon = self.compute_loss(self.activations[len(self.weights)], Y)
                self.biases[l][i] = original_bias
                
                numerical_grads_b[l][i] = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)

            relative_difference_b = np.linalg.norm(grads_b[l] - numerical_grads_b[l]) / (
                np.linalg.norm(grads_b[l]) + np.linalg.norm(numerical_grads_b[l]) + 1e-8
            )
            if relative_difference_b > 1e-6:
                print(f"Gradient check failed for biases in layer {l}: relative difference is {relative_difference_b}")
                return False

        print("Gradient check passed!")
        return True
