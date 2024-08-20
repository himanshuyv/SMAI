import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import time

sys.path.append('./../../')
from models.knn.knn import KNN

from score import Scores

# Load training data
df_train = pd.read_csv("./train.csv")
# df_train = pd.read_csv("./../../data/external/spotify-2/train.csv")
df_train = df_train.drop_duplicates(subset='track_id', keep="first")
df_train = df_train.drop(columns=['Unnamed: 0'])
x_train = df_train.select_dtypes(include=['number'])
# drop columns with name intrumentalness, mode, duration_ms, popularity
# x_train = x_train.drop(columns=['popularity', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature'])
x_train = x_train[['popularity', 'duration_ms', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']]
y_train = df_train['track_genre']

# Load test data
df_test = pd.read_csv("./test.csv")
# df_test = pd.read_csv("./../../data/external/spotify-2/test.csv")
df_test = df_test.drop_duplicates(subset='track_id', keep="first")
df_test = df_test.drop(columns=['Unnamed: 0'])
x_test = df_test.select_dtypes(include=['number'])
# x_test = x_test.drop(columns=['popularity', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature'])
x_test = x_test[['popularity', 'duration_ms', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']]
y_test = df_test['track_genre']

# normalize the data 

x_train = (x_train - x_train.mean()) / x_train.std()
x_test = (x_test - x_test.mean()) / x_test.std()


# Fit KNN model
k = 75
knn = KNN(k)
knn.fit(x_train, y_train)


# taking only 100 samples for faster runtime 
# x_test = x_test.iloc[:1000]
# y_test = y_test.iloc[:1000]

# Measure the runtime of the predict method
start_time_manhattan = time.time()

y_pred = knn.predict(x_test, "manhattan")
best_accuracy_manhattan = np.mean(y_pred == y_test)
# for i in range(50, 250, 25):
#     knn.k = i
#     y_pred = knn.predict(x_test, "manhattan")
#     accuracy = np.mean(y_pred == y_test)
#     best_accuracy_manhattan = max(best_accuracy_manhattan, accuracy)
#     print(f"For i = {i} and mAccuracy: {accuracy:.4f}")


print(f"Best Accuracy Manhattan: {100*best_accuracy_manhattan}%")

end_time_manhattan = time.time() 
run_time_manhattan = end_time_manhattan - start_time_manhattan
print(f"Prediction Runtime: {run_time_manhattan:.4f} seconds")

# start_time_euclidean = time.time()

# y_pred = knn.predict(x_test, "euclidean")
# best_accuracy_euclidean = np.mean(y_pred == y_test)
# # for i in range(50, 250, 25):
# #     knn.k = i
# #     y_pred = knn.predict(x_test, "euclidean")
# #     accuracy = np.mean(y_pred == y_test)
# #     best_accuracy_euclidean = max(best_accuracy_euclidean, accuracy)
# #     print(f"For i = {i} and eAccuracy: {accuracy:.4f}")


# print(f"Best Accuracy Euclidean: {100*best_accuracy_euclidean:.4f}%")

# end_time_euclidean = time.time()
# run_time_euclidean = end_time_euclidean - start_time_euclidean
# print(f"Prediction Runtime: {run_time_euclidean:.4f} seconds")

# Calculate scores
scores = Scores(y_test, y_pred)
print(f"Accuracy: {100*scores.accuracy:.4f}%")
print(f"Micro Precision: {scores.micro_precision:.4f}")
print(f"Micro Recall: {scores.micro_recall:.4f}")
print(f"Micro F1: {scores.micro_f1:.4f}")
print(f"Macro Precision: {scores.macro_precision:.4f}")
print(f"Macro Recall: {scores.macro_recall:.4f}")
print(f"Macro F1: {scores.macro_f1:.4f}")
