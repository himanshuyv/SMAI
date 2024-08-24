import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import imageio


sys.path.append('./../../')
from models.linear_regression.linear_regression import LinearRegression
from performance_measures.regression_score import Scores

# Task 1: Degree 1

# Load data
df = pd.read_csv("./../../data/external/linreg.csv")

# initialize model
model = LinearRegression(learning_rate=0.01)

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
model.fit(x_train, y_train, create_gif=True)

def create_gif(n_images):
    frames = []
    for i in range(n_images):
        img_path = f'./figures/gif_images/{i}.png'
        frame = imageio.imread(img_path)
        frames.append(frame)
    gif_path = './figures/linear_regression.gif'
    imageio.mimsave(gif_path, frames, fps=10)
   
# n_images = model.epock_to_converge // 100
# create_gif(n_images)

y_train = y_train[np.argsort(x_train)]
x_train = np.sort(x_train)

# Plot the regression line
plt.scatter(x_train, y_train)
plt.plot(x_train, model.get_line(x_train), color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regression line')
plt.show()


# Task 2: Degree k

# initialize model

model = LinearRegression(learning_rate=0.01, k=20)

# Fit the model
model.fit(x_train, y_train, create_gif=True)

n_images = model.epock_to_converge // 100
create_gif(n_images)

y_train = y_train[np.argsort(x_train)]
x_train = np.sort(x_train)

# Plot the regression line
plt.scatter(x_train, y_train)
plt.plot(x_train, model.get_line(x_train), color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regression line')
plt.show()


# Tune for Hyperparameter

# Load validation data
x_validate = df_validate['x'].to_numpy()
y_validate = df_validate['y'].to_numpy()

# Fit the model
model.fit(x_validate, y_validate)

min_error = float('inf')
best_k = 0
mse_list = []
for k in range(1, 21):
    model = LinearRegression(learning_rate=0.01, k=k, error_threshold=1e-9)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_validate)
    score = Scores(y_validate, y_pred)
    mse = score.mse
    mse = mse.round(9)
    mse_list.append((mse,k))
    if mse < min_error:
        min_error = mse
        best_k = k

print(f'Best k: {best_k}')
print(f'Minimum MSE: {min_error}')

mse_list = sorted(mse_list)