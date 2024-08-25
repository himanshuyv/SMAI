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
model.fit(x_train, y_train, create_gif=False)

def create_gif(n_images, name):
    frames = []
    for i in range(n_images):
        img_path = f'./figures/gif_images/{i}.png'
        frame = imageio.imread(img_path)
        frames.append(frame)
    imageio.mimsave(name, frames, fps=10)
   
# n_images = model.epoch_to_converge // 100
# create_gif(n_images, "./figures/degree_1.gif")

y_train = y_train[np.argsort(x_train)]
x_train = np.sort(x_train)

# Plot the regression line
fig = plt.figure()
plt.scatter(x_train, y_train)
plt.plot(x_train, model.get_line(x_train), color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regression line')
plt.savefig('./figures/regression_line_degree_1.png')


y_pred_train = model.predict(x_train)
score = Scores(y_train, y_pred_train)
print("Train Metrics for Degree 1")
print(f'MSE: {score.mse}')
print(f'STD: {score.std}')
print(f'Variance: {score.variance}')
print()

# Load test data
x_test = df_test['x'].to_numpy()
y_test = df_test['y'].to_numpy()

y_pred_test = model.predict(x_test)
score = Scores(y_test, y_pred_test)
print("Test Metrics for Degree 1")
print(f'MSE: {score.mse}')
print(f'STD: {score.std}')
print(f'Variance: {score.variance}')
print()

# Task 2: Degree k

# initialize model

model = LinearRegression(learning_rate=0.01, k=20)

# Fit the model
model.fit(x_train, y_train, create_gif=False)

# n_images = model.epoch_to_converge // 100
# create_gif(n_images)

y_train = y_train[np.argsort(x_train)]
x_train = np.sort(x_train)

# Plot the regression line
fig = plt.figure()
plt.scatter(x_train, y_train)
plt.plot(x_train, model.get_line(x_train), color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regression line')
plt.savefig('./figures/regression_line_degree_20.png')


# Tune for Hyperparameter

# Load validation data
x_validate = df_validate['x'].to_numpy()
y_validate = df_validate['y'].to_numpy()

# Fit the model
model.fit(x_validate, y_validate)

# Tune for k
min_error = float('inf')
best_k = 0
mse_list = []
train_list = []
validate_list = []
for k in range(1, 21):
    model = LinearRegression(learning_rate=0.01, k=k, error_threshold=1e-8)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_validate)
    score = Scores(y_validate, y_pred)
    mse = score.mse
    mse = mse.round(9)
    std = score.std.round(9)
    variance = score.variance.round(9)
    mse_list.append((mse,k))
    if mse < min_error:
        min_error = mse
        best_k = k
    
    validate_list.append((mse, std, variance))

    train_mse = model.mse_list[-1].round(9)
    train_std = model.std_list[-1].round(9)
    train_variance = model.variance_list[-1].round(9)
    train_list.append((train_mse, train_std, train_variance))

print(f'Best k: {best_k}')
print(f'Minimum MSE: {min_error}')
print()

# Plot the MSE
fig = plt.figure()
plt.plot([x[1] for x in mse_list], [x[0] for x in mse_list])
plt.xlabel('k')
plt.ylabel('MSE')
plt.title('k vs MSE')
plt.savefig('./figures/k_vs_mse_regression.png')

# Plot mse, std, variance for train and validate
fig , ax = plt.subplots(1, 3, figsize=(15, 5))
x_idx = range(1, 21)
y_mse_train = [x[0] for x in train_list]
y_std_train = [x[1] for x in train_list]
y_variance_train = [x[2] for x in train_list]

y_mse_validate = [x[0] for x in validate_list]
y_std_validate = [x[1] for x in validate_list]
y_variance_validate = [x[2] for x in validate_list]

ax[0].plot(x_idx, y_mse_train)
ax[0].plot(x_idx, y_mse_validate)
ax[0].set_xlabel('k')
ax[0].set_ylabel('MSE')
ax[0].legend(['Train', 'Validate'])

ax[1].plot(x_idx, y_std_train)
ax[1].plot(x_idx, y_std_validate)
ax[1].set_xlabel('k')
ax[1].set_ylabel('STD')
ax[1].legend(['Train', 'Validate'])

ax[2].plot(x_idx, y_variance_train)
ax[2].plot(x_idx, y_variance_validate)
ax[2].set_xlabel('k')
ax[2].set_ylabel('Variance')
ax[2].legend(['Train', 'Validate'])

plt.savefig('./figures/k_vs_mse_std_variance_forTrainTestSplit.png')


# Tune for learning rate
best_k = 19
min_error = float('inf')
best_learning_rate = 0
learning_rate_list = [0.1, 0.01, 0.001, 0.0001]
inference_list = []
for learning_rate in learning_rate_list:
    model = LinearRegression(learning_rate=learning_rate, k=best_k, error_threshold=1e-8)
    model.fit(x_train, y_train)
    n = model.epoch_to_converge
    y_pred = model.predict(x_validate)
    score = Scores(y_validate, y_pred)
    mse = score.mse
    mse = mse.round(9)
    if mse < min_error:
        min_error = mse
        best_learning_rate = learning_rate
    inference_list.append((mse, n))

print(f'Best learning rate: {best_learning_rate}')
print(f'Minimum MSE: {min_error}')
print()

# Plot the epoch ans mse vs learning rate
fig , ax = plt.subplots(1, 2, figsize=(10, 5))
x_idx = [-1, -2, -3, -4]
y_mse = [x[0] for x in inference_list]
y_n = [x[1] for x in inference_list]

ax[0].plot(x_idx, y_mse)
ax[0].set_xlabel('Learning Rate (log scale)')
ax[0].set_ylabel('MSE')

ax[1].plot(x_idx, y_n)
ax[1].set_xlabel('Learning Rate (log scale)')
ax[1].set_ylabel('Epoch to converge')
plt.savefig('./figures/learning_rate_vs_mse_epoch.png')


# Test the model

# Load test data
x_test = df_test['x'].to_numpy()
y_test = df_test['y'].to_numpy()

# Fit the model
model = LinearRegression(learning_rate=best_learning_rate, k=best_k)
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# Evaluate
score = Scores(y_test, y_pred)
print("Test Metrics with best hyperparameters")
print(f'MSE: {score.mse.round(9)}')
print(f'Std: {score.std.round(9)}')
print(f'Variance: {score.variance.round(9)}')

# Create GIF for 5 values of k

k_values = [1, 5, 10, 15, 20]

for k in k_values:
    model = LinearRegression(learning_rate=0.01, k=k)
    model.fit(x_train, y_train, create_gif=True)

    n_images = model.epoch_to_converge // 100
    create_gif(n_images, f'./figures/gif_{k}.gif')

