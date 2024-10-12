import numpy as np
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class MLP_multilabel:
    def __init__(self, learning_rate=0.01, n_epochs=200, batch_size=32, neurons_per_layer=None,
                 activation_function='sigmoid', loss_function='binary_crossentropy', optimizer='sgd', 
                 early_stopping=False, patience=10, gradient_check=False):
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
        self.n_classes = Y.shape[1]  # For multilabel, Y is already in binary format (one-hot encoded)

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

            loss = self.compute_loss(self.predict(X), Y)
            print("Epoch: ", epoch+1, " Loss: ", loss)


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
            weights[i] = np.random.randn(layers[i], layers[i + 1]).astype(np.float64) * 0.1  # Ensure float64
        return weights

    def initialize_biases(self):
        biases = {}
        layers = [self.n_features] + self.neurons_per_layer + [self.n_classes]
        for i in range(1, len(layers)):
            biases[i] = np.random.randn(layers[i]).astype(np.float64) * 0.1  # Ensure float64
        return biases


    def forward_propagation(self, X):
        self.activations = {}
        self.activations[0] = X
        for i in range(1, len(self.weights) + 1):
            z = np.dot(self.activations[i - 1], self.weights[i - 1]) + self.biases[i]
            if i == len(self.weights): 
                self.activations[i] = self.sigmoid(z)  # Sigmoid for multilabel classification
            else:
                self.activations[i] = self.activation(z)

    def sigmoid(self, x):
        x = np.array(x, dtype=np.float64)  # Ensure that x is a numpy array
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

    def activation_derivative(self, x):
        if self.activation_function == 'sigmoid':
            z = self.sigmoid(x)
            return z * (1 - z)
        elif self.activation_function == 'relu':
            return 1. * (x > 0)
        elif self.activation_function == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_function == 'linear':
            return 1

    def backward_propagation(self, Y):
        self.errors = {}
        self.errors[len(self.weights)] = (Y - self.activations[len(self.weights)]).astype(np.float64)  # Ensure float64
        for i in range(len(self.weights) - 1, 0, -1):
            self.errors[i] = np.dot(self.errors[i + 1], self.weights[i].T).astype(np.float64) * self.activation_derivative(self.activations[i])


    def update_weights(self):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i].astype(np.float64)  # Ensure the weights are float64
            self.activations[i] = self.activations[i].astype(np.float64)  # Ensure activations are float64
            self.errors[i + 1] = self.errors[i + 1].astype(np.float64)  # Ensure errors are float64
            
            self.weights[i] += self.learning_rate * np.dot(self.activations[i].T, self.errors[i + 1])
            self.biases[i + 1] += self.learning_rate * np.sum(self.errors[i + 1], axis=0)


    def compute_loss(self, Y_pred, Y_true):
        # Binary Cross Entropy for multilabel classification
        return -np.mean(Y_true * np.log(Y_pred + 1e-8) + (1 - Y_true) * np.log(1 - Y_pred + 1e-8))

    def compute_metrics(self, Y_pred, Y_true):
        # Multilabel classification uses different metrics
        Y_pred = (Y_pred >= 0.5).astype(int)  # Threshold for multilabel prediction
        accuracy = accuracy_score(Y_true, Y_pred)
        precision = precision_score(Y_true, Y_pred, average='macro', zero_division=0)
        recall = recall_score(Y_true, Y_pred, average='macro', zero_division=0)
        f1 = f1_score(Y_true, Y_pred, average='macro', zero_division=0)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def predict(self, X):
        self.forward_propagation(X)
        return (self.activations[len(self.weights)] >= 0.5).astype(int)
