import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys


sys.path.append('./../../')
from models.linear_regression.linear_regression import LinearRegression
from performance_measures.regression_score import Scores

# Task 1: Degree 1

# Load data
df = pd.read_csv("./../../data/external/regularisation.csv")

lambda_ = 0.00001
error_threshold = 1e-8

# initialize model
model = LinearRegression(learning_rate=0.01, k=5, error_threshold=error_threshold)

# Suffle and split the data
df = df.sample(frac=1).reset_index(drop=True)

train_size = int(0.8 * len(df))
validate_size = int(0.1 * len(df))
test_size = len(df) - train_size - validate_size

df_train, df_validate, df_test = np.split(df, [train_size, train_size + validate_size])

# Load training data
x_train = df_train['x'].to_numpy()
y_train = df_train['y'].to_numpy()

# Fit the model
model.fit(x_train, y_train)

y_train = y_train[np.argsort(x_train)]
x_train = np.sort(x_train)

fig = plt.figure()
plt.scatter(x_train, y_train)
plt.plot(x_train, model.get_line(x_train), color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regression line for k=5 and No Regularization')
plt.savefig('./figures/no_regularization_k5.png')

# Load test data
x_test = df_test['x'].to_numpy()
y_test = df_test['y'].to_numpy()

# print("Train Metrics")
# print("k=5, No Regularization")
# print(f'MSE: {model.mse_list[-1]}')
# print(f'STD: {model.std_list[-1]}')
# print(f'Variance: {model.variance_list[-1]}')
# print()

y_pred = model.predict(x_test)
scores = Scores(y_test, y_pred)

print("Test Metrics")
print("k=5, No Regularization")
print(f'MSE: {scores.mse}')
print(f'STD: {scores.std}')
print(f'Variance: {scores.variance}')
print()


model.k = 20

# Fit the model
model.fit(x_train, y_train)

fig = plt.figure()
plt.scatter(x_train, y_train)
plt.plot(x_train, model.get_line(x_train), color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regression line for k=20 and No Regularization')
plt.savefig('./figures/no_regularization_k20.png')

# print("Train Metrics")
# print("k=20, No Regularization")
# print(f'MSE: {model.mse_list[-1]}')
# print(f'STD: {model.std_list[-1]}')
# print(f'Variance: {model.variance_list[-1]}')
# print()

y_pred = model.predict(x_test)
scores = Scores(y_test, y_pred)

print("Test Metrics")
print("k=20, No Regularization")
print(f'MSE: {scores.mse}')
print(f'STD: {scores.std}')
print(f'Variance: {scores.variance}')
print()

# Task 2: Regularization


# initialize model
model = LinearRegression(learning_rate=0.01, k=20, regularization='l1', lambda_=lambda_, error_threshold=error_threshold)

# Fit the model
model.fit(x_train, y_train)

fig = plt.figure()
plt.scatter(x_train, y_train)
plt.plot(x_train, model.get_line(x_train), color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regression line for L1 Regularization')
plt.savefig('./figures/l1_regularization_k20.png')

# print("Train Metrics")
# print("k=20, L1 Regularization")
# print(f'MSE: {model.mse_list[-1]}')
# print(f'STD: {model.std_list[-1]}')
# print(f'Variance: {model.variance_list[-1]}')
# print()

y_pred = model.predict(x_test)
scores = Scores(y_test, y_pred)

print("Test Metrics")
print("k=20, L1 Regularization")
print(f'MSE: {scores.mse}')
print(f'STD: {scores.std}')
print(f'Variance: {scores.variance}')
print()

# initialize model
model = LinearRegression(learning_rate=0.01, k=20, regularization='l2', lambda_= lambda_, error_threshold=error_threshold)

# Fit the model
model.fit(x_train, y_train)

fig = plt.figure()
plt.scatter(x_train, y_train)
plt.plot(x_train, model.get_line(x_train), color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regression line for L2 Regularization')
plt.savefig('./figures/l2_regularization_k20.png')

# print("Train Metrics")
# print("k=20, L2 Regularization")
# print(f'MSE: {model.mse_list[-1]}')
# print(f'STD: {model.std_list[-1]}')
# print(f'Variance: {model.variance_list[-1]}')
# print()

y_pred = model.predict(x_test)
scores = Scores(y_test, y_pred)

print("Test Metrics")
print("k=20, L2 Regularization")
print(f'MSE: {scores.mse}')
print(f'STD: {scores.std}')
print(f'Variance: {scores.variance}')