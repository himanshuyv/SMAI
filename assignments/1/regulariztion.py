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
model = LinearRegression(n_iterations=1000, learning_rate=0.01, k=5, lambda_=0.1)

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

w = model.weights
b = model.bias

def get_line(x, w, b):
    n = len(w)
    y = b
    for i in range(n):
        y += w[i] * x ** (i+1)
    return y


y_train = y_train[np.argsort(x_train)]
x_train = np.sort(x_train)

# Plot the regression line
plt.scatter(x_train, y_train)
plt.plot(x_train, get_line(x_train, w, b), color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regression line')
plt.show()

model.k = 20

# Fit the model
model.fit(x_train, y_train)

w = model.weights
b = model.bias

# Plot the regression line
plt.scatter(x_train, y_train)
plt.plot(x_train, get_line(x_train, w, b), color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regression line')
plt.show()
