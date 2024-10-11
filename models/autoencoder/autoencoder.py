import numpy as np
import sys
sys.path.append('./../../')

from models.mlp.mlp import MLP

class AutoEncoder:
    def __init__(self, input_dim, latent_dim, hidden_layers=[64, 32], activation='relu', optimizer='sgd', epochs=100, learning_rate=0.01, batch_size=32):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.optimizer = optimizer
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        self.encoder = MLP(
            learning_rate=learning_rate, 
            n_epochs=epochs, 
            n_hidden=len(hidden_layers), 
            neurons_per_layer=hidden_layers + [latent_dim], 
            activation_function=activation, 
            loss_function='mean_squared_error',
            optimizer=optimizer
        )
        self.decoder = MLP(
            learning_rate=learning_rate, 
            n_epochs=epochs, 
            n_hidden=len(hidden_layers), 
            neurons_per_layer=[latent_dim] + list(reversed(hidden_layers)) + [input_dim], 
            activation_function=activation, 
            loss_function='mean_squared_error',
            optimizer=optimizer
        )

    def fit(self, X_train, X_val=None):
        self.encoder.fit(X_train, X_train, X_val, X_val)
        latent_rep = self.encoder.predict(X_train)
        self.decoder.fit(latent_rep, X_train, X_val)

    def get_latent(self, X):
        latent_rep = self.encoder.predict(X)
        return latent_rep

    def reconstruct(self, X):
        latent_rep = self.encoder.predict(X)
        reconstructed_X = self.decoder.predict(latent_rep)
        return reconstructed_X
