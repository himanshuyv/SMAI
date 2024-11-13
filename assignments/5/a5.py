import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

import sys
sys.path.append('./../../')
from models.kde.kde import KDE
from models.gmm.gmm import Gmm

import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_data():
    num_samples_large_circle = 3000
    num_samples_small_circle = 500

    theta = np.random.uniform(0, 2*np.pi, num_samples_large_circle)
    r = np.random.uniform(0, 1, num_samples_large_circle)
    x = 2* np.sqrt(r) * np.cos(theta)
    y = 2* np.sqrt(r) * np.sin(theta)
    data_large_circle = np.vstack((x, y)).T
    noise_large_circle = np.random.normal(0, 0.2, (num_samples_large_circle, 2))
    data_large_circle = data_large_circle + noise_large_circle


    theta = np.random.uniform(0, 2*np.pi, num_samples_small_circle)
    r = np.random.uniform(0, 0.2, num_samples_small_circle)
    x = r * np.cos(theta) + 1
    y = r * np.sin(theta) + 1
    data_small_circle = np.vstack((x, y)).T
    data = np.vstack((data_large_circle, data_small_circle))
    noise_small_circle = np.random.normal(0, 0.1, (num_samples_small_circle, 2))
    data_small_circle = data_small_circle + noise_small_circle
    return data

data = generate_synthetic_data()


plt.figure(figsize=(8, 8))
plt.scatter(data[:, 0], data[:, 1], s=5, alpha=0.6, color='black')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.title('KDE Original Data')
plt.grid(True)
plt.savefig('KDE_original_data.png')


kde = KDE(kernel='gaussian', bandwidth=0.5)
kde.fit(data)

point = np.array([1, 1])
density = kde.predict(point)
print(f"Density at {point}: {density}")

point = np.array([0, 0])
density = kde.predict(point)
print(f"Density at {point}: {density}")

kde.visualize(x_range=(-3, 3), y_range=(-3, 3), resolution=100)

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_gmm(data, means, covariances, title, save_path):
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], s=5, alpha=0.6, color="blue", label="Data Points")
    colors = ['red', 'green', 'orange', 'purple', 'cyan']
    
    for i, (mean, cov) in enumerate(zip(means, covariances)):
        color = colors[i % len(colors)]
        eigvals, eigvecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigvecs[0, 1], eigvecs[0, 0]))
        width, height = 2 * np.sqrt(eigvals)
        
        ellip = Ellipse(xy=mean, width=width, height=height, angle=angle, color=color, alpha=0.3)
        plt.gca().add_patch(ellip)
        plt.scatter(mean[0], mean[1], c=color, s=100, marker='x', label=f"Component {i+1}")
    
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.savefig(save_path)


for k in [2, 5, 10]:
    gmm_model = Gmm(k=k, n_iter=100)
    gmm_model.fit(data)

    pi, mu, sigma = gmm_model.get_params()
    plot_gmm(data, mu, sigma, f"GMM with {k} components", f"GMM_{k}_components.png")
    
plt.show()