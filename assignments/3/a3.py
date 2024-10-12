import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import MultiLabelBinarizer
import wandb

import sys
sys.path.append('./../../')

from models.mlp.mlp import MLP
from models.mlp_multilabel.MLP_multilabel import MLP_multilabel
from models.mlp_regression.regression import MLPR
from models.autoencoder.autoencoder import AutoEncoder





def MLP_singleLabel(train_sweep=False):
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
    plt.savefig('./figures/dataAnalysis2.png')

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    Y = Y.to_numpy()

    X_train = X[:int(0.8*len(X))]
    Y_train = Y[:int(0.8*len(Y))]
    X_validation = X[int(0.8*len(X)):int(0.9*len(X))]
    Y_validation = Y[int(0.8*len(Y)):int(0.9*len(Y))]
    X_test = X[int(0.9*len(X)):]
    Y_test = Y[int(0.9*len(Y)):]

    mlp = MLP(n_epochs=1000, neurons_per_layer=[64,32], activation_function='relu', optimizer='mini-batch', batch_size=32, learning_rate=0.05)

    Y_train = Y_train
    mlp.fit(X_train, Y_train)
    Y_pred = mlp.predict(X_test)
    metrics = mlp.compute_metrics(Y_test, Y_pred)
    print("Accuracy: ", metrics['accuracy'])
    print("Precision: ", metrics['precision'])
    print("Recall: ", metrics['recall'])
    print("F1: ", metrics['f1'])
    print("Loss: ", mlp.compute_loss(X, Y))

    mlp.gradient_check(X_train, Y_train)

    if train_sweep:
        sweep_config = {
            'method': 'grid',
            'metric': {
                'name': 'val_accuracy',
                'goal': 'maximize'
            },
            'parameters': {
                'batch_size': {
                    'values': [64]
                },

                'learning_rate': {
                    'values': [0.01,0.05,0.1]
                },
                'activation_function': {
                    'values': ['relu', 'sigmoid', 'tanh']
                },
                'optimizer': {
                    'values': ['sgd', 'mini-batch', 'batch']
                },
                'neurons_per_layer': {
                    'values': [[64], [64, 32], [64, 32, 16]]
                }
            }
        }

        def train_sweep(config=None):
            with wandb.init(config=config):
                config = wandb.config
                mlp = MLP(n_epochs=1000,
                          learning_rate=config.learning_rate, 
                          neurons_per_layer=config.neurons_per_layer, 
                          activation_function=config.activation_function, 
                          optimizer=config.optimizer,
                          batch_size=config.batch_size)
                mlp.fit(X_train, Y_train, X_validation, Y_validation)

        sweep_id = wandb.sweep(sweep_config, project='mlp-classifier-sweep-3')
        wandb.agent(sweep_id, train_sweep)

def MLP_multiLabel():
    df = pd.read_csv('./../../data/external/advertisement.csv')
    def encode_y(advertisement):
        advertisement['labels'] = advertisement['labels'].str.split()
        binarizer = MultiLabelBinarizer()
        vecs = binarizer.fit_transform(advertisement['labels'])
        Y = pd.DataFrame(vecs, columns=binarizer.classes_)
        return Y

    def encode_x(advertisement):
        df.drop(columns=['labels'], inplace=True)
        df.drop(columns=['city'], inplace=True)
        most_bought_item = list(advertisement['most bought item'].unique())
        occupation = list(advertisement['occupation'].unique())
        married = list(advertisement['married'].unique())
        education = list(advertisement['education'].unique())
        df['most bought item'] = df['most bought item'].map( lambda x: most_bought_item.index(x))
        df['occupation'] = df['occupation'].map( lambda x: occupation.index(x))
        df['married'] = df['married'].map( lambda x: married.index(x))
        df['education'] = df['education'].map( lambda x: education.index(x))
        df['gender'] = df['gender'].map( lambda x: 1 if x == 'MALE' else 0)
        return df

    Y = encode_y(df)
    X = encode_x(df)
    X = X.to_numpy()
    X = X.astype(np.float64)
    X_mean = X.mean()
    X_std = X.std()
    X = (X - X_mean) / X_std
    # print(X)
    Y = Y.to_numpy()
    X_train = X[:int(0.8*len(X))]
    Y_train = Y[:int(0.8*len(Y))]
    X_validation = X[int(0.8*len(X)):int(0.9*len(X))]
    Y_validation = Y[int(0.8*len(Y)):int(0.9*len(Y))]
    X_test = X[int(0.9*len(X)):]
    Y_test = Y[int(0.9*len(Y)):]
    mlp = MLP_multilabel(n_epochs=1000, neurons_per_layer=[48, 16], activation_function='tanh', optimizer='mini-batch', batch_size=16, learning_rate=0.05)
    mlp.fit(X_train, Y_train)
    Y_pred = mlp.predict(X_train)
    metrics = mlp.compute_metrics(Y_pred, Y_train)
    
    print("Accuracy: ", metrics['accuracy'])
    print("Precision: ", metrics['precision'])
    print("Recall: ", metrics['recall'])
    print("F1: ", metrics['f1'])
    
def MLP_regression(train_sweep=False):
    df = pd.read_csv('./../../data/external/HousingData.csv')
    df = df.dropna()

    df_mean = df.mean()
    df_std = df.std()
    df_min = df.min()
    df_max = df.max()

    print("\nMean: \n", df_mean)
    print("\nStandard Deviation: \n", df_std)
    print("\nMin: \n", df_min)
    print("\nMax: \n", df_max)

    df.hist()
    plt.tight_layout()
    plt.savefig('./figures/dataAnalysis3.png')

    X = df.drop(columns=['MEDV'])
    Y = df['MEDV']
    standard_scaler = StandardScaler()
    X_standardized = pd.DataFrame(standard_scaler.fit_transform(X), columns=X.columns)
    min_max_scaler = MinMaxScaler()
    X_normalized = pd.DataFrame(min_max_scaler.fit_transform(X_standardized), columns=X.columns)

    X = X_normalized.to_numpy()
    Y = Y.to_numpy()
 
    X_train = X[:int(0.8*len(X))]
    Y_train = Y[:int(0.8*len(Y))]
    X_validation = X[int(0.8*len(X)):int(0.9*len(X))]
    Y_validation = Y[int(0.8*len(Y)):int(0.9*len(Y))]
    X_test = X[int(0.9*len(X)):]
    Y_test = Y[int(0.9*len(Y)):]

    mlp_reg = MLPR(
        learning_rate=0.001,
        n_epochs=1000,
        batch_size=32,
        neurons_per_layer=[64, 32],
        activation_function='relu',
        optimizer='mini-batch'
    )

    mlp_reg.fit(X_train, Y_train.reshape(-1, 1))
    Y_pred = mlp_reg.predict(X_test)
    mse = mlp_reg.compute_loss(Y_pred, Y_test)
    print(f"Mean Squared Error on Test Set: {mse:.4f}")

    def train_sweep(config=None):
        with wandb.init(config=config):
            config = wandb.config
            mlp_reg = MLPR(
                learning_rate=config.learning_rate,
                n_epochs=1000,
                batch_size=config.batch_size,
                neurons_per_layer=config.neurons_per_layer,
                activation_function=config.activation_function,
                optimizer=config.optimizer
            )
            mlp_reg.fit(X_train, Y_train.reshape(-1, 1), X_validation, Y_validation.reshape(-1, 1))

    if train_sweep:
        sweep_config = {
            'method': 'grid',
            'metric': {
                'name': 'val_accuracy',
                'goal': 'maximize'
            },
            'parameters': {
                'batch_size': {
                    'values': [64]
                },

                'learning_rate': {
                    'values': [0.01,0.05,0.1]
                },
                'activation_function': {
                    'values': ['relu', 'sigmoid', 'tanh']
                },
                'optimizer': {
                    'values': ['sgd', 'mini-batch', 'batch']
                },
                'neurons_per_layer': {
                    'values': [[64], [64, 32], [64, 32, 16]]
                }
            }
        }

        sweep_id = wandb.sweep(sweep_config, project='mlp-regression')
        wandb.agent(sweep_id, train_sweep)

def auto_encoder():
    # pass
    df = pd.read_csv("./../../data/external/spotify.csv")
    df = df.drop(columns=['Unnamed: 0'])
    df = df.drop_duplicates(subset='track_id', keep="first")

    df = df.sample(frac=1).reset_index(drop=True)
    df_numerical = df.select_dtypes(include=['number'])

    def normalize(df):
        return (df - df.min()) / (df.max() - df.min())
    df_numerical = normalize(df_numerical)

    X = df_numerical
    Y = df['track_genre']

    X = X.to_numpy()

    autoencoder = AutoEncoder(input_dim=X.shape[1], latent_dim=8, hidden_layers=[64, 32], activation='relu', optimizer='sgd', epochs=100, learning_rate=0.01, batch_size=32)
    autoencoder.fit(X)

    latent_rep = autoencoder.get_latent(X)
    reconstructed_X = autoencoder.reconstruct(X)

    print("Latent Representation: ", latent_rep)
    print("Reconstructed X: ", reconstructed_X)





# MLP_singleLabel(train_sweep=False)
MLP_multiLabel()
# MLP_regression(train_sweep=True)
# auto_encoder()