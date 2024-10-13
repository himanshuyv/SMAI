import numpy as np
import sys
sys.path.append('./../../')

from models.mlp.regression import MLPR

class AutoEncoder:
    def __init__(self, input_dim, latent_dim, learning_rate=0.01, n_epochs=200, batch_size=32, 
                 neurons_per_layer=None, activation_function='relu', loss_function='mean_squared_error', 
                 optimizer='sgd', early_stopping=False, patience=10):
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        autoencoder_layers = neurons_per_layer[:-1] + [latent_dim] + neurons_per_layer[::-1] + [input_dim]
        self.autoencoder = MLPR(learning_rate=learning_rate, n_epochs=n_epochs, batch_size=batch_size, 
                                neurons_per_layer=autoencoder_layers, activation_function=activation_function, 
                                loss_function=loss_function, optimizer=optimizer, early_stopping=early_stopping, 
                                patience=patience)

    def fit(self, X):
        self.autoencoder.fit(X, X)

    def get_latent(self, X):
        self.autoencoder.forward_propagation(X)
        activations = self.autoencoder.activations
        latent_rep = activations[len(self.autoencoder.neurons_per_layer) // 2]
        return latent_rep

    def reconstruct(self, X):
        return self.autoencoder.predict(X)
