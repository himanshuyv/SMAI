import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

sys.path.append('./../../')
from models.knn.knn import KNN
from models.pca.pca import PCA
from performance_measures.knn_score import Scores

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

n = X.shape[1]

pca = PCA(n_components = n)

pca.fit(X)

explained_var = pca.getExplainedVariance()
plt.figure()
plt.plot(range(1, len(explained_var) + 1), explained_var, marker = 'o')
plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.title('Explained Variance')
plt.savefig("./figures/pca_knn_explained_variance.png")

plt.figure()
plt.plot(range(1, len(pca.eig_values)+1), pca.eig_values, marker = 'o')
plt.xlabel('Number of components')
plt.ylabel('Eigenvalues fraction')
plt.title('Eigenvalues Fraction')
plt.savefig("./figures/pca_knn_eigenvalues_fraction.png")

pca = PCA(n_components = 10)
pca.fit(X)
X1 = pca.transform(X)

train_size = int(0.8 * len(df))

x_train = X1[:train_size]
y_train = Y[:train_size]
x_validate = X1[train_size:]
y_validate = Y[train_size:]

k = 20
knn = KNN(k)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_validate, distance_metric='manhattan')

score = Scores(y_validate, y_pred)
accuracy = score.accuracy
print(f'Accuracy: {accuracy}')
print(f'Micro Precision: {score.micro_precision}')
print(f'Micro Recall: {score.micro_recall}')
print(f'Micro F1: {score.micro_f1}')
print(f'Macro Precision: {score.macro_precision}')
print(f'Macro Recall: {score.macro_recall}')
print(f'Macro F1: {score.macro_f1}')


