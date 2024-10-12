import numpy as np
import sys
sys.path.append('./../../')

from models.mlp.mlp import MLP

class AutoEncoder:
    def __init__(self, input_dim, hidden_dims, latent_dim, learning_rate=0.01, n_epochs=200, batch_size=32, activation_function='sigmoid', optimizer='sgd', early_stopping=False, patience=10):
        self.encoder = MLP(learning_rate=learning_rate,
                           n_epochs=n_epochs,
                           batch_size=batch_size,
                           neurons_per_layer=hidden_dims + [latent_dim],
                           activation_function=activation_function,
                           loss_function='mean_squared_error',
                           optimizer=optimizer,
                           early_stopping=early_stopping,
                           patience=patience)

        self.decoder = MLP(learning_rate=learning_rate,
                           n_epochs=n_epochs,
                           batch_size=batch_size,
                           neurons_per_layer=[latent_dim] + hidden_dims[::-1] + [input_dim],
                           activation_function=activation_function,
                           loss_function='mean_squared_error',
                           optimizer=optimizer,
                           early_stopping=early_stopping,
                           patience=patience)

    def fit(self, X_train, X_val=None):
        self.encoder.fit(X_train, X_train) 
        latent_rep = self.encoder.predict(X_train)
        self.decoder.fit(latent_rep, X_train, X_val=X_val) 
    
    def get_latent(self, X):
        return self.encoder.predict(X)

    def reconstruct(self, X):
        latent_rep = self.get_latent(X)
        return self.decoder.predict(latent_rep)
