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

# initialize model
model = LinearRegression(n_Epochs=1000, learning_rate=0.01, k=5)

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

y_pred = model.predict(x_train)
scores = Scores(y_train, y_pred)
print("k=5, No Regularization")
print(f'MSE: {scores.mse}')
print(f'STD: {scores.std}')
print(f'Variance: {scores.variance}')
print()

# Plot the regression line
plt.scatter(x_train, y_train)
plt.plot(x_train, model.get_line(x_train), color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regression line for k=5 and No Regularization')
plt.show()

model.k = 20

# Fit the model
model.fit(x_train, y_train)

y_pred = model.predict(x_train)
scores = Scores(y_train, y_pred)
print("k=20, No Regularization")
print(f'MSE: {scores.mse}')
print(f'STD: {scores.std}')
print(f'Variance: {scores.variance}')
print()

# Plot the regression line
plt.scatter(x_train, y_train)
plt.plot(x_train, model.get_line(x_train), color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regression line for k=20 and No Regularization')
plt.show()

# Task 2: Regularization

# initialize model
model = LinearRegression(n_Epochs=1000, learning_rate=0.01, k=20, regularization='l1', alpha=0.01)

# Fit the model
model.fit(x_train, y_train)

y_pred = model.predict(x_train)
scores = Scores(y_train, y_pred)
print("k=20, L1 Regularization")
print(f'MSE: {scores.mse}')
print(f'STD: {scores.std}')
print(f'Variance: {scores.variance}')
print()

# Plot the regression line
plt.scatter(x_train, y_train)
plt.plot(x_train, model.get_line(x_train), color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regression line for L1 Regularization')
plt.show()

# initialize model
model = LinearRegression(n_Epochs=1000, learning_rate=0.01, k=20, regularization='l2', alpha=0.01)

# Fit the model
model.fit(x_train, y_train)

y_pred = model.predict(x_train)
scores = Scores(y_train, y_pred)
print("k=20, L2 Regularization")
print(f'MSE: {scores.mse}')
print(f'STD: {scores.std}')
print(f'Variance: {scores.variance}')

# Plot the regression line
plt.scatter(x_train, y_train)
plt.plot(x_train, model.get_line(x_train), color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regression line for L2 Regularization')
plt.show()
