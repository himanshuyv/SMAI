import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import wandb

import sys
sys.path.append('./../../')

from models.mlp.mlp import MLP
from performance_measures.knn_score import Scores

sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
		
		'batch_size': {
			'values': [1,32,1000]
        },

        'learning_rate': {
            'values': [0.01,0.1,0.05,0.5]
        },
        'activation_function': {
            'values': ['relu', 'sigmoid', 'tanh']
        },
        'optimizer': {
            'values': ['sgd']
        },
        'neurons_per_layer': {
            'values': [[64], [64, 32], [64, 32, 16]]
        }
    }
}




df = pd.read_csv('./../../data/external/WineQT.csv')
df = df.drop(columns=['Id'])
print(df.shape)

df_mean = df.mean()
df_std = df.std()
df_min = df.min()
df_max = df.max()

print("\nMean: \n", df_mean)
print("\nStandard Deviation: \n", df_std)
print("\nMin: \n", df_min)
print("\nMax: \n", df_max)

X = df.drop(columns=['quality'])
Y = df['quality']

X.hist()
plt.tight_layout()
# plt.savefig('./../../figures/3/dataAnalysis2.png')

scaler = StandardScaler()
X = scaler.fit_transform(X)

Y = Y.to_numpy()

X_train = X[:int(0.8*len(X))]
Y_train = Y[:int(0.8*len(Y))]
X_validation = X[int(0.8*len(X)):int(0.9*len(X))]
Y_validation = Y[int(0.8*len(Y)):int(0.9*len(Y))]
X_test = X[int(0.9*len(X)):]
Y_test = Y[int(0.9*len(Y)):]

mlp = MLP(n_epochs=1000, n_hidden=2, neurons_per_layer=[64,32], activation_function='sigmoid', loss_function='mean_squared_error', optimizer='sgd', batch_size=32)

Y_train = Y_train
mlp.fit(X_train, Y_train)

Y_pred = mlp.predict(X_test)
Y_pred = Y_pred
print(Y_pred)
print('y_test:', Y_test)
scores = Scores(Y_test, Y_pred)
print("Accuracy: ", scores.accuracy)

def train_sweep(config=None):
    with wandb.init(config=config):
        config = wandb.config
        mlp = MLP(n_epochs=1000,
                  n_hidden=3,
                  learning_rate=config.learning_rate, 
                  neurons_per_layer=config.neurons_per_layer, 
                  activation_function=config.activation_function, 
                  optimizer=config.optimizer,
                  batch_size=config.batch_size)
        mlp.fit(X_train, Y_train, X_validation, Y_validation)

sweep_id = wandb.sweep(sweep_config, project='mlp-classifier-sweep-2')
wandb.agent(sweep_id, train_sweep)