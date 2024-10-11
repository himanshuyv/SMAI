import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
import wandb

import sys
sys.path.append('./../../')

from models.mlp.mlp import MLP
from models.mlp_multilabel.mlp import MLP_multilabel
from models.mlp_regression.regression import MLPR





def MLP_sigleLabel(train_sweep=False):
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

    mlp = MLP(n_epochs=1000, n_hidden=2, neurons_per_layer=[64,32], activation_function='sigmoid', loss_function='mean_squared_error', optimizer='sgd', batch_size=32, learning_rate=0.01)

    Y_train = Y_train
    mlp.fit(X_train, Y_train)
    Y_pred = mlp.predict(X_test)
    metrics = mlp.compute_metrics(Y_test, Y_pred)
    print("Accuracy: ", metrics['accuracy'])
    print("Precision: ", metrics['precision'])
    print("Recall: ", metrics['recall'])
    print("F1: ", metrics['f1'])
    loss = mlp.compute_loss(Y_pred, Y_test)
    print("Loss: ", loss)
    
    if train_sweep:
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


def MLP_multiLabel():
    df = pd.read_csv('./../../data/external/advertisement.csv')
    def encode_data(advertisement):
        advertisement['labels'] = advertisement['labels'].str.split()
        gender = pd.get_dummies(advertisement['gender'], prefix='gender')
        advertisement = pd.concat([advertisement, gender], axis=1)
        occupation = pd.get_dummies(advertisement['occupation'], prefix='occupation')
        advertisement = pd.concat([advertisement, occupation], axis=1)
        binarizer = MultiLabelBinarizer()
        vecs = binarizer.fit_transform(advertisement['labels'])
        binarized_df = pd.DataFrame(vecs, columns=binarizer.classes_)
        advertisement = pd.concat([advertisement, binarized_df], axis=1)
        advertisement.drop(['labels', 'most bought item', 'city', 'gender', 'occupation'], axis=1, inplace=True)
        advertisement['married'] = advertisement['married'].apply(lambda x: 1 if x == True else 0)
        advertisement['education'] = advertisement['education'].apply(lambda x: 1 if x == 'High School' else 2 if x == 'Bachelor' else 3 if x == 'Master' else 4)
        return advertisement
    df_encoded = encode_data(df)
    # save the encoded data
    df_encoded.to_csv('./../../data/interim/advertisement_encoded.csv', index=False)

    X = df_encoded.drop(columns=['beauty', 'books', 'clothing', 'electronics', 'food', 'furniture', 'home', 'sports'])
    Y = df_encoded[['beauty', 'books', 'clothing', 'electronics', 'food', 'furniture', 'home', 'sports']]
    X = X.to_numpy()
    # X_min = X.min()
    # X_max = X.max()
    X_mean = X.mean()
    X_std = X.std()
    X = (X - X_mean) / X_std
    Y = Y.to_numpy()
    print('Y:',Y)
    X_train = X[:int(0.8*len(X))]
    Y_train = Y[:int(0.8*len(Y))]
    X_validation = X[int(0.8*len(X)):int(0.9*len(X))]
    Y_validation = Y[int(0.8*len(Y)):int(0.9*len(Y))]
    X_test = X[int(0.9*len(X)):]
    Y_test = Y[int(0.9*len(Y)):]
    print('Y_train:',Y_train)
    mlp = MLP_multilabel(n_epochs=100, n_hidden=2, neurons_per_layer=[48,16], activation_function='sigmoid', optimizer='sgd', batch_size=32, learning_rate=0.5)
    mlp.fit(X_train, Y_train)
    Y_pred = mlp.predict(X_test)
    metrics = mlp.compute_metrics(Y_pred, Y_test)
    
    print("Accuracy: ", metrics['accuracy'])
    print("Precision: ", metrics['precision'])
    print("Recall: ", metrics['recall'])
    print("F1: ", metrics['f1'])
    
def MLP_regression():
    df = pd.read_csv('./../../data/external/HousingData.csv')
    df = df.dropna()

    X = df.drop(columns=['MEDV'])
    Y = df['MEDV']

    X = X.to_numpy()
    Y = Y.to_numpy()

    X_train = X[:int(0.8*len(X))]
    Y_train = Y[:int(0.8*len(Y))]
    X_validation = X[int(0.8*len(X)):int(0.9*len(X))]
    Y_validation = Y[int(0.8*len(Y)):int(0.9*len(Y))]
    X_test = X[int(0.9*len(X)):]
    Y_test = Y[int(0.9*len(Y)):]

    mlp_reg = MLPR(learning_rate=0.01, n_epochs=500, n_hidden=3, neurons_per_layer=[64, 32, 16], activation_function='relu', optimizer='sgd')
    mlp_reg.fit(X_train, Y_train)

    Y_pred = mlp_reg.predict(X_test)
    loss = mlp_reg.compute_loss(Y_pred, Y_test)
    print("Loss: ", loss)

# MLP_sigleLabel(train_sweep=False)
# MLP_multiLabel()
MLP_regression()