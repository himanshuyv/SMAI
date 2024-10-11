import numpy as np
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class MLP_multilabel:
    def __init__(self, learning_rate=0.01, n_epochs=1000, n_hidden=2, batch_size=32, neurons_per_layer=None,
                 activation_function='sigmoid', loss_function='cross_entropy', optimizer='sgd', early_stopping=False):
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
        self.X = X.astype(np.float64)
        self.Y = Y.astype(np.float64)
        self.n_samples, self.n_features = X.shape
        self.n_classes = Y.shape[1]

        self.weights = self.initialize_weights()
        self.biases = self.initialize_biases()

        stop_counter = 0
        for epoch in range(self.n_epochs):
            for i in range(0, self.n_samples, self.batch_size):
                X_batch = X[i:i+self.batch_size].astype(np.float64)
                Y_batch = Y[i:i+self.batch_size].astype(np.float64)
                self.forward_propagation(X_batch)
                self.backward_propagation(Y_batch)
                self.update_weights()
            Y_train_pred = self.predict(X)
            train_loss = self.compute_loss(Y_train_pred, Y)
            print(f'loss: {train_loss}')
            if (X_val is not None and Y_val is not None):
                Y_val_pred = self.predict(X_val)
                Y_train_pred = self.predict(X)
                val_loss = self.compute_loss(Y_val_pred, Y_val)
                train_loss = self.compute_loss(Y_train_pred, Y)

                score_train = self.compute_metrics(Y_train_pred, Y)
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
                

    def initialize_weights(self):
        weights = {}
        layers = [self.n_features] + self.neurons_per_layer + [self.n_classes]
        for i in range(len(layers) - 1):
            weights[i] = np.random.randn(layers[i], layers[i + 1]).astype(np.float64) * 0.1
        return weights

    def initialize_biases(self):
        biases = {}
        layers = [self.n_features] + self.neurons_per_layer + [self.n_classes]
        for i in range(1, len(layers)):
            biases[i] = np.random.randn(layers[i]).astype(np.float64) * 0.1
        return biases

    def forward_propagation(self, X):
        self.activations = {}
        self.activations[0] = X.astype(np.float64)
        for i in range(1, len(self.weights)+1):
            z = np.dot(self.activations[i - 1], self.weights[i - 1]) + self.biases[i]
            if i == len(self.weights):
                self.activations[i] = self.sigmoid(z).astype(np.float64)
            else:
                self.activations[i] = self.activation(z).astype(np.float64)

    def activation(self, x):
        x = np.array(x, dtype=np.float64)
        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == 'relu':
            return np.maximum(0, x)
        elif self.activation_function == 'tanh':
            return np.tanh(x)
        elif self.activation_function == 'linear':
            return x

    def sigmoid(self, x):
        x = x.astype(np.float64)
        return 1 / (1 + np.exp(-x))

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
        self.errors[len(self.weights)] = (self.activations[len(self.weights)] - Y).astype(np.float64) * self.activation_derivative(self.activations[len(self.weights)])
        for i in range(len(self.weights) - 1, 0, -1):
            self.errors[i] = (np.dot(self.errors[i + 1], self.weights[i].T) * self.activation_derivative(self.activations[i])).astype(np.float64)

    def update_weights(self):
        for i in range(len(self.weights)):
            if self.optimizer == 'sgd':
                self.weights[i] -= self.learning_rate * np.dot(self.activations[i].T, self.errors[i + 1]).astype(np.float64)
                self.biases[i + 1] -= self.learning_rate * np.sum(self.errors[i + 1], axis=0).astype(np.float64)

    def compute_loss(self, Y_pred, Y_true):
        if self.loss_function == 'cross_entropy':
            Y_pred = np.clip(Y_pred, 1e-15, 1 - 1e-15)
            loss = -np.sum((Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred)), axis=1)
            return np.mean(loss)
        elif self.loss_function == 'mean_squared_error':
            return np.mean((Y_true - Y_pred) ** 2)


    def compute_metrics(self, Y_pred, Y_true):
        pass
        print(Y_pred)
        # print(Y_true)
        Y_pred_bin = (Y_pred >= 0.5).astype(int)
        metrics = {
            'accuracy': accuracy_score(Y_true, Y_pred_bin),
            'precision': precision_score(Y_true, Y_pred_bin, average='macro'),
            'recall': recall_score(Y_true, Y_pred_bin, average='macro'),
            'f1': f1_score(Y_true, Y_pred_bin, average='macro')
        }
        return metrics

    def predict(self, X):
        self.forward_propagation(X)
        output = self.activations[len(self.weights)]
        return output
