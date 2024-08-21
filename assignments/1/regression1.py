import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

sys.path.append('./../../')
from models.linear_regression.linear_regression import LinearRegression
from performance_measures.regression_score import Scores

# Task 1: Degree 1

# Load data
df = pd.read_csv("./../../data/external/linreg.csv")

# Plotting data
plt.scatter(df['x'], df['y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter plot of X and Y')
plt.show()

# initialize model
model = LinearRegression(n_iterations=1000, learning_rate=0.01)

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

# Predict
y_pred = model.predict(x_train)

# Plot the regression line
plt.scatter(x_train, y_train)
plt.plot(x_train, y_pred, color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regression line')
plt.show()

# Calculate scores
scores = Scores(y_train, y_pred)

print(f"Mean Squared Error: {scores.mse}")


# Task 2: Degree k

# initialize model

model = LinearRegression(n_iterations=10000, learning_rate=0.01, k=10)

# Fit the model
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_train)

# sort values
y_pred = y_pred[np.argsort(x_train)]
y_train = y_train[np.argsort(x_train)]
x_train = np.sort(x_train)

# Plot the regression line
plt.scatter(x_train, y_train)
plt.plot(x_train, y_pred, color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regression line')
plt.show()

# Calculate scores
scores = Scores(y_train, y_pred)

print(f"Mean Squared Error: {scores.mse}")

