import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import librosa
import os
import glob
import seaborn as sns
from sklearn.model_selection import train_test_split
from hmmlearn import hmm
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

import sys
sys.path.append('./../../')
from models.kde.kde import KDE
from models.gmm.gmm import Gmm
from models.rnn.rnn import BitCounterRNN
from models.rnn.rnn import BitCountingDataset, collate_fn

def kde_fun():
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
    plt.savefig('./figures/KDE_original_data.png')


    kde = KDE(kernel='gaussian', bandwidth=0.5)
    kde.fit(data)

    point = np.array([1, 1])
    density = kde.predict(point)
    print(f"Density at {point}: {density}")

    point = np.array([0, 0])
    density = kde.predict(point)
    print(f"Density at {point}: {density}")

    kde.visualize(x_range=(-3, 3), y_range=(-3, 3), resolution=100)

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
        plot_gmm(data, mu, sigma, f"GMM with {k} components", f"./figures/GMM_{k}_components.png")
        
    plt.show()

def hmm_fun():
    def extract_mfcc(file_path, n_mfcc=13):
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        return mfccs.T 

    dataset_path = './../../data/external/recordings'
    data = {str(i): [] for i in range(10)}

    for file_path in glob.glob(os.path.join(dataset_path, '*.wav')):
        digit = file_path.replace('\\','/').split('/')[-1][0]
        mfcc_features = extract_mfcc(file_path)
        data[digit].append(mfcc_features)

    def plot_mfcc(mfcc, title='MFCC'):
        sns.heatmap(mfcc.T, cmap='viridis')
        plt.title(title)
        plt.ylabel('MFCC Coefficients')
        plt.xlabel('Time')

    for digit in data:
        plt.figure(figsize=(8, 8))
        for i, mfcc in enumerate(data[digit][:3]):
            plt.subplot(3, 1, i+1)
            plot_mfcc(mfcc, title=f'Digit {digit} - Sample {i+1}')
        plt.tight_layout()  
        plt.savefig(f'./figures/Digit_{digit}_MFCC.png')

    models = {}

    for digit, features in data.items():
        X = np.concatenate(features)
        lengths = [len(f) for f in features]
        model = hmm.GaussianHMM(n_components=5, covariance_type="diag", n_iter=100)
        model.fit(X, lengths)
        models[digit] = model

    def predict_digit(mfcc):
        log_likelihoods = {}
        for digit, model in models.items():
            log_likelihood = model.score(mfcc)
            log_likelihoods[digit] = log_likelihood
        return max(log_likelihoods, key=log_likelihoods.get)

    def evaluate_accuracy(data):
        correct = 0
        total = 0
        for digit, features in data.items():
            for mfcc in features:
                prediction = predict_digit(mfcc)
                if prediction == digit:
                    correct += 1
                total += 1
        accuracy = correct / total
        return accuracy

    data_list = [(digit, mfcc) for digit, features in data.items() for mfcc in features]

    train_list, test_list = train_test_split(data_list, test_size=0.2)

    train_data = {str(i): [] for i in range(10)}
    test_data = {str(i): [] for i in range(10)}

    for digit, mfcc in train_list:
        train_data[digit].append(mfcc)
    for digit, mfcc in test_list:
        test_data[digit].append(mfcc)

    accuracy = evaluate_accuracy(test_data)
    print(f"Recognition Accuracy on Test Set: {accuracy * 100:.2f}%")


def rnn_fun():
    def generate_bit_count_data(num_samples=100000, max_len=16):
        data = []
        labels = []
        for _ in range(num_samples):
            length = np.random.randint(1, max_len + 1)
            sequence = np.random.randint(0, 2, length)
            label = np.sum(sequence)
            data.append(sequence)
            labels.append(label)
        return data, labels

    def evaluate(model, data_loader, criterion, device):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for sequences, labels, lengths in data_loader:
                sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.to(device)
                outputs = model(sequences, lengths)
                loss = criterion(outputs.squeeze(), labels)
                total_loss += loss.item() * sequences.size(0)
        return total_loss / len(data_loader.dataset)

    data, labels = generate_bit_count_data()
    train_data, temp_data, train_labels, temp_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
    val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=42)

    train_dataset = BitCountingDataset(train_data, train_labels)
    val_dataset = BitCountingDataset(val_data, val_labels)
    test_dataset = BitCountingDataset(test_data, test_labels)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = BitCounterRNN(input_size=1, hidden_size=32, num_layers=1)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for sequences, labels, lengths in train_loader:
            sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.to(device)
            outputs = model(sequences, lengths)
            loss = criterion(outputs.squeeze(), labels.squeeze(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * sequences.size(0)
        
        train_loss = total_train_loss / len(train_loader.dataset)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")


    def generate_out_of_distribution_data(start_len=17, end_len=32, samples_per_length=1000):
        data = []
        labels = []
        for length in range(start_len, end_len + 1):
            for _ in range(samples_per_length):
                sequence = np.random.randint(0, 2, length)
                label = np.sum(sequence)
                data.append(sequence)
                labels.append(label)
        return data, labels

    ood_data, ood_labels = generate_out_of_distribution_data()
    ood_dataset = BitCountingDataset(ood_data, ood_labels)
    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    lengths = list(range(17, 33))
    mae_per_length = []

    for length in lengths:
        length_data = [seq for seq in ood_data if len(seq) == length]
        length_labels = [label for seq, label in zip(ood_data, ood_labels) if len(seq) == length]
        
        length_dataset = BitCountingDataset(length_data, length_labels)
        length_loader = DataLoader(length_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        mae = evaluate(model, length_loader, criterion, device)
        mae_per_length.append(mae)

    plt.figure(figsize=(10, 6))
    plt.plot(lengths, mae_per_length, marker='o', color='b')
    plt.title('Model Generalization Across Sequence Lengths')
    plt.xlabel('Sequence Length')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.grid(True)
    plt.savefig('./figures/RNN_generalization.png')
    plt.show()





# kde_fun()
# hmm_fun()
# rnn_fun()

if __name__ == '__main__':
    while True:
        print("1. KDE")
        print("2. HMM")
        print("3. RNN")
        print("4. Exit")
        choice = int(input("Enter your choice: "))
        if choice == 1:
            kde_fun()
        elif choice == 2:
            hmm_fun()
        elif choice == 3:
            rnn_fun()
        elif choice == 4:
            break
        else:
            print("Invalid choice. Please enter again.")