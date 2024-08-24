import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
import time
from sklearn.neighbors import KNeighborsClassifier

sys.path.append('./../../')
from models.knn.knn import KNN
from performance_measures.knn_score import Scores

# # Task 1

# # Load data
# df = pd.read_csv("./../../data/external/spotify.csv")
# df = df.drop(columns=['Unnamed: 0'])
# df = df.drop_duplicates(subset='track_id', keep="first")
# df_numerical = df.select_dtypes(include=['number'])

# # normalizing data
# def normalize(df):
#     return (df - df.min()) / (df.max() - df.min())
# df_numerical = normalize(df_numerical)

# # Plotting data

# # Historgram
# # df_numerical.hist(bins=50, figsize=(15, 10))
# # plt.show()

# # Boxplot
# # df_numerical.boxplot(figsize=(15, 10))
# # plt.show()

# # Violin plot
# # sns.violinplot(data=df_numerical)
# # plt.show()

# # Pairwise Scatterplot
# # sns.pairplot(df_numerical)
# # plt.show()


# # Task 3

# # suffle the data
# df = df.sample(frac=1).reset_index(drop=True)

# # split the data
# train_size = int(0.8 * len(df))
# validate_size = int(0.1 * len(df))
# test_size = len(df) - train_size - validate_size

# df = df.drop(columns=['key', 'mode'])
# df_train, df_validate, df_test = np.split(df, [train_size, train_size + validate_size])


# # Load training data
# x_train = df_train.select_dtypes(include=['number'])
# # x_train.drop_duplicates(subset='track_id', keep='first')
# y_train = df_train['track_genre']

# # Load test data
# x_test = df_test.select_dtypes(include=['number'])
# # x_test.drop_duplicates(subset='track_id', keep='first')
# y_test = df_test['track_genre']

# # Load validation data
# x_validate = df_validate.select_dtypes(include=['number'])
# # x_validate.drop_duplicates(subset='track_id', keep='first')
# y_validate = df_validate['track_genre']

# # Normalize data
# x_train = normalize(x_train)
# x_test = normalize(x_test)
# x_validate = normalize(x_validate)

# # Fit KNN model
# k = 30
# knn = KNN(k)
# knn.fit(x_train, y_train)

# knn_sklearn = KNeighborsClassifier(n_neighbors=k)
# knn_sklearn.fit(x_train, y_train)

# # taking only 1000 samples for faster runtime 
# # x_validate = x_validate.iloc[:10]
# # y_validate = y_validate.iloc[:10]

# start_myknn = time.time()
# print("Predicting")
# y_pred = knn.predict(x_validate, distance_metric='cosine')
# print(np.mean(y_pred == y_validate))
# end_myknn = time.time()
# print(f"Time taken by my KNN: {end_myknn - start_myknn}")

# start_sklearn = time.time()
# y_pred_sklearn = knn_sklearn.predict(x_validate)
# print(np.mean(y_pred_sklearn == y_validate))
# end_sklearn = time.time()
# print(f"Time taken by sklearn KNN: {end_sklearn - start_sklearn}")

# # tune for k
# def tune_k(k_values, x_validate, y_validate, distance_metric):
#     accuracy_values = []
#     for k in k_values:
#         knn.k = k
#         y_pred = knn.predict(x_validate, distance_metric=distance_metric)
#         accuracy = np.mean(y_pred == y_validate)
#         accuracy = int(accuracy * 100)
#         accuracy_values.append((accuracy, (k, distance_metric)))
#     return accuracy_values

# k_values = range(10, 100, 10)

# # manhattan tuning for k
# accuracy_values_manhattan = tune_k(k_values, x_validate, y_validate, 'manhattan')
    
# # euclidean tuning for k
# accuracy_values_euclidean = tune_k(k_values, x_validate, y_validate, 'euclidean')

# # cosine tuning for k
# accuracy_values_cosine = tune_k(k_values, x_validate, y_validate, 'cosine')


# all_pairs = accuracy_values_manhattan + accuracy_values_euclidean + accuracy_values_cosine

# all_pairs.sort(reverse=True)
# print("Top 10 pairs")
# print(all_pairs[:10])

# # k vs accuracy plot

# def plot_k_vs_accuracy(accuracy_values_pairs):
#     distance_metric = accuracy_values_pairs[0][1][1]
#     accuracy_values = [x[0] for x in accuracy_values_pairs]
#     k_values = [x[1][0] for x in accuracy_values_pairs]
#     plt.plot(k_values, accuracy_values)
#     plt.xlabel('k')
#     plt.ylabel('accuracy')
#     plt.title(f'k vs accuracy for {distance_metric}')
#     plt.show()

# # plot_k_vs_accuracy(accuracy_values_manhattan)
# # plot_k_vs_accuracy(accuracy_values_euclidean)
# # plot_k_vs_accuracy(accuracy_values_cosine)

# # Test for best k and distance metric
# best_k = all_pairs[0][1][0]
# best_distance_metric = all_pairs[0][1][1]

# knn.k = best_k
# y_pred = knn.predict(x_test, distance_metric=best_distance_metric)
# accuracy = np.mean(y_pred == y_test)
# print(f"Accuracy for test data for best k: {best_k} and distance metric: {best_distance_metric} is {accuracy*100}%")

# # Calculate scores
# scores = Scores(y_test, y_pred)
# print(f"Accuracy: {scores.accuracy}")
# print(f"Micro Precision: {scores.micro_precision}")
# print(f"Macro Precision: {scores.macro_precision}")
# print(f"Micro Recall: {scores.micro_recall}")
# print(f"Macro Recall: {scores.macro_recall}")
# print(f"Micro F1 Score: {scores.micro_f1}")
# print(f"Macro F1 Score: {scores.macro_f1}")

# Second data set
# Load data
df_train = pd.read_csv("./../../data/external/spotify-2/train.csv")
df_validate = pd.read_csv("./../../data/external/spotify-2/validate.csv")
df_test = pd.read_csv("./../../data/external/spotify-2/test.csv")

df_train = df_train.drop_duplicates(subset='track_id', keep='first')
df_validate = df_validate.drop_duplicates(subset='track_id', keep='first')
df_test = df_test.drop_duplicates(subset='track_id', keep='first')

df_train.drop(columns=['Unnamed: 0'])
df_validate.drop(columns=['Unnamed: 0'])
df_test.drop(columns=['Unnamed: 0'])

df_train = df_train.drop(columns=['key', 'mode'])
df_validate = df_validate.drop(columns=['key', 'mode'])
df_test = df_test.drop(columns=['key', 'mode'])

x_train = df_train.select_dtypes(include=['number'])
y_train = df_train['track_genre']

x_validate = df_validate.select_dtypes(include=['number'])
y_validate = df_validate['track_genre']

x_test = df_test.select_dtypes(include=['number'])
y_test = df_test['track_genre']

best_k = 60
best_distance_metric = 'euclidean'
knn = KNN(60)
knn.k = best_k
knn.fit(x_train, y_train)
y_pred = knn.predict(x_validate, distance_metric=best_distance_metric)
accuracy = np.mean(y_pred == y_validate)

print(f"Accuracy for validation data for best k: {best_k} and distance metric: {best_distance_metric} is {accuracy*100}%")

# Calculate scores
scores = Scores(y_validate, y_pred)
print(f"Accuracy: {scores.accuracy}")
print(f"Micro Precision: {scores.micro_precision}")
print(f"Macro Precision: {scores.macro_precision}")
print(f"Micro Recall: {scores.micro_recall}")
print(f"Macro Recall: {scores.macro_recall}")
print(f"Micro F1 Score: {scores.micro_f1}")
print(f"Macro F1 Score: {scores.macro_f1}")
