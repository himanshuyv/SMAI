import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
import wandb

import sys
sys.path.append('./../../')

from models.mlp.mlp import MLP
from models.mlp.MLP_multilabel import MLP_multilabel
from models.mlp.regression import MLPR
from models.autoencoder.autoencoder import AutoEncoder
from models.knn.knn import KNN
from models.mlp.mlp_merged import MLP_merged
from performance_measures.knn_score import Scores


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

    if (len(np.unique(Y)) > 2):
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        Y_one_hot = one_hot_encoder.fit_transform(Y.reshape(-1, 1))
    else:
        Y_one_hot = Y

    X_train = X[:int(0.8*len(X))]
    Y_train = Y[:int(0.8*len(Y))]
    Y_train_one_hot = Y_one_hot[:int(0.8*len(Y_one_hot))]
    X_validation = X[int(0.8*len(X)):int(0.9*len(X))]
    Y_validation = Y[int(0.8*len(Y)):int(0.9*len(Y))]
    Y_validation_one_hot = Y_one_hot[int(0.8*len(Y_one_hot)):int(0.9*len(Y_one_hot))]
    X_test = X[int(0.9*len(X)):]
    Y_test = Y[int(0.9*len(Y)):]
    Y_test_one_hot = Y_one_hot[int(0.9*len(Y_one_hot)):]

    # mlp = MLP(n_epochs=1000, neurons_per_layer=[64,32], activation_function='relu', optimizer='batch', batch_size=128, learning_rate=0.005)
    mlp = MLP_merged(n_epochs=1000, neurons_per_layer=[64,32], activation_function='relu', optimizer='batch', loss_function='cross_entropy',batch_size=128, learning_rate=0.005)
    mlp.fit(X_train, Y_train_one_hot)

    mlp.gradient_check(X_train, Y_train_one_hot)
    
    Y_pred = mlp.predict(X_test)
    Y_pred_label = one_hot_encoder.inverse_transform(Y_pred)
    # metrics = mlp.compute_metrics(Y_pred_label, Y_test)
    metrics = mlp.compute_metrics_classification(Y_pred_label, Y_test)
    print("Test Metrics: ")
    print("Accuracy: ", metrics['accuracy'])
    print("Precision: ", metrics['precision'])
    print("Recall: ", metrics['recall'])
    print("F1: ", metrics['f1'])
    print("Loss: ", mlp.compute_loss(X_test, Y_test_one_hot))

    if train_sweep:
        sweep_config = {
            'method': 'grid',
            'metric': {
                'name': 'val_accuracy',
                'goal': 'maximize'
            },
            'parameters': {
                'batch_size': {
                    'values': [128, 32]
                },

                'learning_rate': {
                    'values': [0.001,0.005,0.01]
                },
                'activation_function': {
                    'values': ['relu', 'sigmoid', 'tanh', 'linear']
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
                          batch_size=config.batch_size,
                          early_stopping=True)
                mlp.fit(X_train, Y_train_one_hot, X_validation, Y_validation_one_hot)

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
    mlp = MLP_multilabel(n_epochs=1000, neurons_per_layer=[64, 32], activation_function='tanh', optimizer='mini-batch', batch_size=16, learning_rate=0.1)
    mlp.fit(X_train, Y_train)
    Y_pred = mlp.predict(X_test)
    accuracy = np.mean(Y_pred == Y_test)
    metrics = mlp.compute_metrics(Y_pred, Y_test)
    print("Soft Accuracy: ", accuracy)
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
    Y = Y.reshape(-1, 1)
 
    X_train = X[:int(0.8*len(X))]
    Y_train = Y[:int(0.8*len(Y))]
    X_validation = X[int(0.8*len(X)):int(0.9*len(X))]
    Y_validation = Y[int(0.8*len(Y)):int(0.9*len(Y))]
    X_test = X[int(0.9*len(X)):]
    Y_test = Y[int(0.9*len(Y)):]

    # mlp_reg = MLPR(learning_rate=0.001, n_epochs=1000, batch_size=16, neurons_per_layer=[64, 32, 16], activation_function='relu', optimizer='sgd')
    mlp_reg = MLP_merged(learning_rate=0.001, n_epochs=1000, batch_size=16, neurons_per_layer=[64, 32, 16], loss_function="mean_squared_error",activation_function='relu', optimizer='sgd', is_classification=False)

    mlp_reg.fit(X_train, Y_train)
    mlp_reg.gradient_check(X_train, Y_train)
    Y_pred = mlp_reg.predict(X_test)
    # metrics = mlp_reg.compute_metrics(Y_pred, Y_test)
    metrics = mlp_reg.compute_metrics_regression(Y_pred, Y_test)
    print("Test Metrics: ")
    print("MSE: ", metrics['mse'])
    print("RMSE: ", metrics['rmse'])
    print("MAE: ", metrics['mae'])
    print("R2: ", metrics['r_squared'])


    if train_sweep:
        sweep_config = {
            'method': 'grid',
            'metric': {
                'name': 'val_rmse',
                'goal': 'minimize'
            },
            'parameters': {
                'batch_size': {
                    'values': [64, 32, 16]
                },

                'learning_rate': {
                    'values': [0.001, 0.01, 0.005]
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
                mlp_reg = MLPR(
                    learning_rate=config.learning_rate,
                    n_epochs=1000,
                    batch_size=config.batch_size,
                    neurons_per_layer=config.neurons_per_layer,
                    activation_function=config.activation_function,
                    optimizer=config.optimizer,
                    early_stopping=True
                )
                mlp_reg.fit(X_train, Y_train.reshape(-1, 1), X_validation, Y_validation.reshape(-1, 1))

        sweep_id = wandb.sweep(sweep_config, project='mlp-regression')
        wandb.agent(sweep_id, train_sweep)

def LogisticRegression():
    df = pd.read_csv('./../../data/external/diabetes.csv')
    df = df.dropna()
    X = df.drop(columns=['Outcome'])
    Y = df['Outcome']
    X = X.to_numpy()
    Y = Y.to_numpy().reshape(-1, 1)

    X = (X - X.mean()) / X.std()

    m_mse = MLPR(learning_rate=0.01, n_epochs=1000, batch_size=32, neurons_per_layer=[64, 32], activation_function='relu', optimizer='mini-batch', loss_function='mean_squared_error')
    m_mse.fit(X, Y)
    mse_loss_list = m_mse.loss_list

    m_bce = MLPR(learning_rate=0.01, n_epochs=1000, batch_size=32, neurons_per_layer=[64, 32], activation_function='relu', optimizer='mini-batch', loss_function='binary_cross_entropy')
    m_bce.fit(X, Y)
    bce_loss_list = m_bce.loss_list

    plt.plot(mse_loss_list, label='Mean Squared Error')
    plt.plot(bce_loss_list, label='Binary Cross Entropy')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.savefig('./figures/logistic_regression.png')

def auto_encoder():
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
    Y = Y.to_numpy()

    one_hot_encoder = OneHotEncoder(sparse_output=False)
    Y_one_hot = one_hot_encoder.fit_transform(Y.reshape(-1, 1))

    X_train = X[:int(0.8*len(X))]
    Y_train = Y[:int(0.8*len(Y))]
    Y_train_one_hot = Y_one_hot[:int(0.8*len(Y_one_hot))]
    X_validation = X[int(0.8*len(X)):]
    Y_validation = Y[int(0.8*len(Y)):]
    Y_validation_one_hot = Y_one_hot[int(0.8*len(Y_one_hot)):]

    autoencoder = AutoEncoder(input_dim=X.shape[1], latent_dim=8, neurons_per_layer=[64, 32, 16], activation_function='relu', optimizer='sgd', n_epochs=200, learning_rate=0.01, batch_size=32)
    autoencoder.fit(X_train)

    X_train_reduced = autoencoder.get_latent(X_train)
    X_validation_reduced = autoencoder.get_latent(X_validation)
    print(X_train_reduced.shape)
    print(X_validation_reduced.shape)

    knn = KNN(k=20)
    knn.fit(X_train_reduced, Y_train)
    Y_pred = knn.predict(X_validation_reduced, 'manhattan')
    scores = Scores(Y_pred, Y_validation)
    print("Accuracy: ", scores.accuracy)
    print("Micro Precision: ", scores.micro_precision)
    print("Micro Recall: ", scores.micro_recall)
    print("Micro F1: ", scores.micro_f1)

    print("Macro Precision: ", scores.macro_precision)
    print("Macro Recall: ", scores.macro_recall)
    print("Macro F1: ", scores.macro_f1)



    
    mlp = MLP(n_epochs=500, neurons_per_layer=[48, 16], activation_function='tanh', optimizer='mini-batch', batch_size=32, learning_rate=0.01)
    mlp.fit(X_train_reduced, Y_train_one_hot)
    Y_pred_one_hot = mlp.predict(X_validation_reduced)
    Y_pred = one_hot_encoder.inverse_transform(Y_pred_one_hot)
    scores = Scores(Y_pred, Y_validation)
    print("Accuracy: ", scores.accuracy)
    print("Micro Precision: ", scores.micro_precision)
    print("Micro Recall: ", scores.micro_recall)
    print("Micro F1: ", scores.micro_f1)
    print("Macro Precision: ", scores.macro_precision)
    print("Macro Recall: ", scores.macro_recall)
    print("Macro F1: ", scores.macro_f1)
    print("Loss: ", mlp.compute_loss(X_validation_reduced, Y_validation_one_hot))


# MLP_singleLabel(train_sweep=False)
# MLP_multiLabel()
# MLP_regression(train_sweep=False)
# LogisticRegression()
# auto_encoder()

while True:
    print("1. MLP Single Label")
    print("2. MLP Multi Label")
    print("3. MLP Regression")
    print("4. Logistic Regression")
    print("5. Auto Encoder")
    print("6. Exit")
    choice = int(input("Enter your choice: "))
    if choice == 1:
        sweep = input("Do you want to train sweep? (y/n): ")
        if sweep == 'y':
            MLP_singleLabel(train_sweep=True)
        else:
            MLP_singleLabel()
    elif choice == 2:
        MLP_multiLabel()
    elif choice == 3:
        sweep = input("Do you want to train sweep? (y/n): ")
        if sweep == 'y':
            MLP_regression(train_sweep=True)
        else:
            MLP_regression()
    elif choice == 4:
        LogisticRegression()
    elif choice == 5:
        auto_encoder()
    elif choice == 6:
        break
    else:
        print("Invalid Choice")