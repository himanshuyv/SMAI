import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('./../../')

from models.mlp.mlp import MLP
from performance_measures.knn_score import Scores


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

mlp = MLP(n_epochs=100, n_hidden=4, neurons_per_layer=[64,64,64,64], activation_function='relu', loss_function='mean_squared_error', optimizer='sgd')

Y_train = Y_train - min(Y_train)
mlp.fit(X_train, Y_train)

Y_pred = mlp.predict(X_test)
Y_pred = Y_pred + 3
print(Y_pred)
print('y_test:', Y_test)
scores = Scores(Y_test, Y_pred)
print("Accuracy: ", scores.accuracy)
