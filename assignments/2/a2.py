import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import sys
sys.path.append('./../../')

from models.kmeans.kmeans import KMeans
from models.gmm.gmm import Gmm
from models.pca.pca import PCA
from models.knn.knn import KNN
from performance_measures.knn_score import Scores



# load data
df_big = pd.read_feather('./../../data/external/word-embeddings.feather')
X = df_big['vit'].to_numpy()
X_big = np.array([x for x in X])
Y_big = df_big['words'].to_numpy()

df_small = pd.read_csv("./../../data/external/clustering.csv")
X_small = df_small[['x','y']].to_numpy()
Y_small = df_small['color'].to_numpy()

K_KMEANS1_SMALL = 3
K_KMEANS1 = 11
K_GMM1 = 2
K2 = 4
K_KMEANS3 = 8
K_GMM3 = 4
PCA_DIM = 130

def kmeans_fun(k_, X, Y, type="small"):
    kmeans = KMeans(k = k_, n_iter = 100)
    kmeans.fit(X)
    cost = kmeans.getCost()
    print(f'Cost: {cost}')

    centroids = kmeans.centroids
    labels = kmeans.labels

    if (type == "small"):
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
        plt.title('Kmeans')
        plt.savefig('./figures/kmeans_small.png')
        
    clusters = {}
    for i in range(k_):
        clusters[i] = []

    for i in range(len(Y)):
        clusters[labels[i]].append(Y[i])

    for i in range(k_):
        print(f'Cluster {i+1}: {clusters[i]}')

    wcss = []
    for i in range(1, 20):
        kmeans = KMeans(k = i, n_iter = 100)
        kmeans.fit(X)
        wcss.append(kmeans.getCost())

    figure = plt.figure()
    plt.plot(range(1, 20), wcss, marker = 'o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig(f'./figures/kmeans_elbow_method_{type}.png')


def gmm_fun(k_, X, Y, type="small"):
    gmm = Gmm(k = k_, n_iter = 10)
    gmm.fit(X)

    print("Likelihood: ", gmm.getLikelihood())
    print("pi: ", gmm.get_params()[0])

    # ploting the clusters
    gamma = gmm.getMembership()
    clusters = np.argmax(gamma, axis = 1)

    if (type == "small"):
        
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c = clusters, cmap = 'viridis')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('GMM Clustering')
        plt.savefig("./figures/gmm_clusters.png")
        return

    AIC = []
    BIC = []

    for i in range(1, 11):
        gmm = Gmm(k = i, n_iter = 10)
        gmm.fit(X)
        AIC.append(gmm.aic())
        BIC.append(gmm.bic())
        print("likelihood: ", gmm.getLikelihood())

    plt.figure()
    plt.plot(range(1, 11), AIC, marker = 'o', label = 'AIC')
    plt.plot(range(1, 11), BIC, marker = 'o', label = 'BIC')
    plt.xlabel('Number of clusters')
    plt.ylabel('Information Criterion')
    plt.title('AIC and BIC for GMM')
    plt.legend()
    plt.savefig("./figures/gmm_aic_bic_mine.png")

    # Inbuilt GMM
    gmm = GaussianMixture(n_components = 10, max_iter = 10)
    gmm.fit(X)

    print("Inbuilt GMM Likelihood: ", gmm.score(X))


    AIC_inbuilt = []
    BIC_inbuilt = []

    for i in range(1, 11):
        gmm = GaussianMixture(n_components = i, max_iter = 100)
        gmm.fit(X)
        likelihood = gmm.score(X)
        AIC_inbuilt.append(gmm.aic(X))
        BIC_inbuilt.append(gmm.bic(X))

    plt.figure()
    plt.plot(range(1, 11), AIC_inbuilt, marker = 'o', label = 'AIC')
    plt.plot(range(1, 11), BIC_inbuilt, marker = 'o', label = 'BIC')
    plt.xlabel('Number of clusters')
    plt.ylabel('Information Criterion')
    plt.title('AIC and BIC for GMM')
    plt.legend()
    plt.savefig("./figures/gmm_aic_bic_inbuilt.png")

def pca_fun(X):
    pca = PCA(n_components = 2)
    pca.fit(X)

    print("Before Transformation:", pca.checkPCA(X))
    print(X.shape)
    X1 = pca.transform(X)
    X1 = np.real(X1)
    print("After PCA")
    print(X1.shape)

    plt.figure()
    plt.scatter(X1[:, 0], X1[:, 1])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('PCA - 2D Projection')
    plt.savefig("./figures/pca_2d.png")


    pca = PCA(n_components = 3)

    pca.fit(X)

    print("Before Transformation:", pca.checkPCA(X))
    print(X.shape)
    X2 = pca.transform(X)
    X2 = np.real(X2)
    print("After PCA")
    print(X2.shape)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X2[:, 0], X2[:, 1], X2[:, 2], c='b', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('PCA - 3D Projection')
    plt.tight_layout()
    plt.savefig("./figures/pca_3d.png")


def pca_clustering_fun(X, Y):
    pca = PCA(n_components = 2)
    pca.fit(X)

    X1 = pca.transform(X)
    X1 = np.real(X1)

    kmeans = KMeans(k = K2, n_iter = 100)
    kmeans.fit(X1)

    centroids = kmeans.centroids
    labels = kmeans.labels

    plt.figure()
    plt.scatter(X1[:, 0], X1[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
    plt.title('Kmeans')
    plt.savefig('./figures/pca_kmeans.png')

    pca = PCA(n_components = 200)
    pca.fit(X)

    explained_var = pca.getExplainedVariance()

    plt.figure()
    plt.plot(range(1, len(explained_var) + 1), explained_var)
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance')
    plt.title('Explained Variance')
    plt.savefig("./figures/pca_explained_variance.png")

    n = 10

    global PCA_DIM
    for i in range(1, len(explained_var)+1):
        if explained_var[i] > 0.95:
            PCA_DIM = i
            print(f"Number of components for 95% variance: {i}")
            break

    plt.figure()
    plt.plot(range(1, n+1), pca.eig_values[:n], marker = 'o')
    plt.xlabel('Number of components')
    plt.ylabel('Eigenvalues fraction')
    plt.title('Eigenvalues Fraction')
    plt.savefig("./figures/pca_eigenvalues_fraction.png")

    pca = PCA(n_components = PCA_DIM)
    pca.fit(X)
    X2 = pca.transform(X)
    X2 = np.real(X2)

    wcss = []
    for i in range(1, 20):
        kmeans = KMeans(k = i, n_iter = 100)
        kmeans.fit(X2)
        wcss.append(kmeans.getCost())

    plt.figure()
    plt.plot(range(1, 20), wcss, marker = 'o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig('./figures/pca_kmeans_elbow.png')

    kmeans = KMeans(k = K_KMEANS3, n_iter = 100)
    kmeans.fit(X2)
    cost = kmeans.getCost()
    print(f'Cost of Kmeans for reduced Dataset: {cost}')

    gmm = Gmm(k = K2, n_iter = 10)
    gmm.fit(X1)

    clusters = np.argmax(gmm.getMembership(), axis = 1)

    plt.figure()
    plt.scatter(X1[:, 0], X1[:, 1], c = clusters, cmap = 'viridis')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('GMM Clustering')
    plt.savefig("./figures/pca_gmm_clusters.png")

    AIC = []
    BIC = []

    for i in range(1, 11):
        gmm = Gmm(k = i, n_iter = 10)
        gmm.fit(X2)
        AIC.append(gmm.aic())
        BIC.append(gmm.bic())

    plt.figure()
    plt.plot(range(1, 11), AIC, marker = 'o', label = 'AIC')
    plt.plot(range(1, 11), BIC, marker = 'o', label = 'BIC')
    plt.xlabel('Number of clusters')
    plt.ylabel('Information Criterion')
    plt.legend()
    plt.title('AIC and BIC for GMM')
    plt.savefig("./figures/pca_gmm_aic_bic.png")

    AIC_inbuilt = []
    BIC_inbuilt = []

    for i in range(1, 11):
        gmm = GaussianMixture(n_components = i, max_iter = 100)
        gmm.fit(X2)
        AIC_inbuilt.append(gmm.aic(X2))
        BIC_inbuilt.append(gmm.bic(X2))

    plt.figure()
    plt.plot(range(1, 11), AIC_inbuilt, marker = 'o', label = 'AIC')
    plt.plot(range(1, 11), BIC_inbuilt, marker = 'o', label = 'BIC')
    plt.xlabel('Number of clusters')
    plt.ylabel('Information Criterion')
    plt.title('AIC and BIC for GMM')
    plt.legend()
    plt.savefig("./figures/pca_gmm_aic_bic_inbuilt.png")

    gmm = Gmm(k = K_GMM3, n_iter = 10)
    gmm.fit(X2)
    print("Likelihood for reduced Dataset: ", gmm.getLikelihood())
    

def hierarchical_fun(X, Y):
    linkage_methods = ['single', 'complete', 'average', 'ward', 'weighted']
    dendrograms = {}

    for method in linkage_methods:
        if method == 'ward':
            distance_methods = ['euclidean']
        else:
            distance_methods = ['euclidean', 'cosine']
        for distance in distance_methods:
            Z = linkage(X, method=method, metric=distance)
            dendrograms[f"{method}_{distance}"] = Z
            plt.figure()
            dendrogram(Z)
            plt.title(f"Dendrogram for {method} linkage and {distance} distance")
            plt.tight_layout()
            plt.savefig(f"./figures/dendrogram_{method}_{distance}.png")


    best_method = 'complete'
    best_distance = 'euclidean'


    def print_clusters(kbest):
        Z = dendrograms[f"{best_method}_{best_distance}"]
        clusters = fcluster(Z, kbest, criterion='maxclust')
        clusters_list = [] 
        for i in range(1, kbest+1):
            cluster = [Y[j] for j in range(len(clusters)) if clusters[j] == i]
            clusters_list.append(cluster)
        for i in range(kbest):
            print(f"Cluster {i+1}: {clusters_list[i]}")


    kbest1 = K_KMEANS1
    kbest2 = K_GMM1

    print("\nKmeans best k\n")
    print_clusters(kbest1)

    print("\nGmm best k\n")
    print_clusters(kbest2)

def pca_knn_fun():
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

subtask_lists = ["3: Kmeans", "4: GMM", "5: PCA", "6: PCA+Clustering", "7: Comparison", "8: Hierarchical", "9: PCA+KNN", "10: Exit"]

while True:
    print("\n\nChoose a subtask:")
    for task in subtask_lists:
        print(task)

    subtask = int(input("\nEnter the subtask number: "))

    if subtask == 3:
        print("Choose a dataset: 1: Small, 2: Big")
        dataset = int(input())
        if dataset == 1:
            kmeans_fun(K_KMEANS1_SMALL, X_small, Y_small)
        elif dataset == 2:
            kmeans_fun(K_KMEANS1, X_big, Y_big, "big")

    elif subtask == 4:
        print("Choose a dataset: 1: Small, 2: Big")
        dataset = int(input("Enter the dataset number: "))
        if dataset == 1:
            gmm_fun(K_GMM1, X_small, Y_small)
        elif dataset == 2:
            gmm_fun(K_GMM1, X_big, Y_big, "big")

    elif subtask == 5:
        pca_fun(X_big)

    elif subtask == 6:
        pca_clustering_fun(X_big, Y_big)

    elif subtask == 7:
        pass

    elif subtask == 8:
        hierarchical_fun(X_big, Y_big)

    elif subtask == 9:
        pca_knn_fun()

    elif subtask == 10:
        break

    else:
        print("Invalid subtask number")