import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

sys.path.append('./../../')
from models.knn.knn import KNN
from models.pca.pca import PCA

# Task 1

# Load data
df = pd.read_csv("./../../data/external/spotify.csv")
df = df.drop(columns=['Unnamed: 0'])
df = df.drop_duplicates(subset='track_id', keep="first")

# suffle the data
df = df.sample(frac=1).reset_index(drop=True)
df_numerical = df.select_dtypes(include=['number'])

# normalizing data
def normalize(df):
    return (df - df.min()) / (df.max() - df.min())
df_numerical = normalize(df_numerical)


X = df_numerical
Y = df['track_genre']

n = X.shape[1]

pca = PCA(n_components = n)

pca.fit(X)

explained_var = pca.getExplainedVariance()
plt.plot(range(1, len(explained_var) + 1), explained_var, marker = 'o')
plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.show()

plt.plot(range(1, len(pca.eig_values)+1), pca.eig_values, marker = 'o')
plt.xlabel('Number of components')
plt.ylabel('Eigenvalues fraction')
plt.show()

pca = PCA(n_components = 10)
pca.fit(X)
X1 = pca.transform(X)

train_size = int(0.8 * len(df))
validate_size = int(0.1 * len(df))
test_size = len(df) - train_size - validate_size

x_train, x_validate, x_test = np.split(X1, [train_size, train_size + validate_size])
y_train, y_validate, y_test = np.split(Y, [train_size, train_size + validate_size])

x_train = normalize(x_train)
x_test = normalize(x_test)
x_validate = normalize(x_validate)

k = 21
knn = KNN(k)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_validate, distance_metric='manhattan')
print(np.mean(y_pred == y_validate))