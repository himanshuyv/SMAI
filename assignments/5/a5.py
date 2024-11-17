import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import glob
import seaborn as sns
from hmmlearn import hmm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from nltk.corpus import words


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

import sys
sys.path.append('./../../')
from models.kde.kde import KDE
from models.gmm.gmm import Gmm
from models.rnn.rnn import BitCounterRNN, BitCountingDataset, collate_fn_rnn
from models.ocr.ocr import OCRModel, OCRDataset

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
        r = np.random.uniform(0, 0.25, num_samples_small_circle)
        x = r * np.cos(theta) + 1
        y = r * np.sin(theta) + 1
        data_small_circle = np.vstack((x, y)).T
        data = np.vstack((data_large_circle, data_small_circle))
        noise_small_circle = np.random.normal(0, 0.01, (num_samples_small_circle, 2))
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

    kde = KDE(kernel='triangular', bandwidth=1)
    kde.fit(data)
    kde.visualize(x_range=(-3, 3), y_range=(-3, 3), resolution=100)

    kde = KDE(kernel='box', bandwidth=1)
    kde.fit(data)
    kde.visualize(x_range=(-3, 3), y_range=(-3, 3), resolution=100)
    
    kde = KDE(kernel='gaussian', bandwidth=1)
    kde.fit(data)
    kde.visualize(x_range=(-3, 3), y_range=(-3, 3), resolution=100)


    kde = KDE(kernel='triangular', bandwidth=0.5)
    kde.fit(data)
    kde.visualize(x_range=(-3, 3), y_range=(-3, 3), resolution=100)

    kde = KDE(kernel='box', bandwidth=0.5)
    kde.fit(data)
    kde.visualize(x_range=(-3, 3), y_range=(-3, 3), resolution=100)

    kde = KDE(kernel='gaussian', bandwidth=0.5)
    kde.fit(data)
    kde.visualize(x_range=(-3, 3), y_range=(-3, 3), resolution=100)

    # point = np.array([1, 1])
    # density = kde.predict(point)
    # print(f"Density at {point}: {density}")

    # point = np.array([0, 0])
    # density = kde.predict(point)
    # print(f"Density at {point}: {density}")

    for k in [2, 5, 10]:
        gmm_model = Gmm(k=k, n_iter=100)
        gmm_model.fit(data)

        memberships = gmm_model.getMembership()
        memberships = np.argmax(memberships, axis=1)

        plt.figure(figsize=(8, 8))
        plt.scatter(data[:, 0], data[:, 1], s=5, alpha=0.6, c=memberships, cmap='viridis')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.title(f'GMM Clustering with K={k}')
        plt.grid(True)
        plt.savefig(f'./figures/GMM_K_{k}.png')

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
        plt.figure(figsize=(12, 8))
        plot_mfcc(data[digit][0], title=f'Digit {digit}')
        plt.tight_layout()  
        plt.savefig(f'./figures/Digit_{digit}_MFCC.png')

    models = {}

    for digit, features in data.items():
        X = np.concatenate(features)
        lengths = []
        for feature in features:
            lengths.append(feature.shape[0])
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

    data_list = []
    for digit, features in data.items():
        for mfcc in features:
            data_list.append((digit, mfcc))

    train_size = int(0.8 * len(data_list))
    train_list, test_list = data_list[:train_size], data_list[train_size:]

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
        data, labels = [], []
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
                loss = criterion(outputs.squeeze(), labels.squeeze(-1))
                total_loss += loss.item() * sequences.size(0)
        return total_loss / len(data_loader.dataset)

    data, labels = generate_bit_count_data()
    train_size = int(0.8 * len(data))
    val_size = int(0.1 * len(data))
    
    train_data, val_data, test_data = data[:train_size], data[train_size:train_size + val_size], data[train_size + val_size:]
    train_labels, val_labels, test_labels = labels[:train_size], labels[train_size:train_size + val_size], labels[train_size + val_size:]

    train_dataset = BitCountingDataset(train_data, train_labels)
    val_dataset = BitCountingDataset(val_data, val_labels)
    test_dataset = BitCountingDataset(test_data, test_labels)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_rnn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_rnn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_rnn)

    model = BitCounterRNN(input_size=1, hidden_size=32, num_layers=1)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_epochs = 10

    def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            train_progress = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
            
            for sequences, labels, lengths in train_progress:
                sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.to(device)
                outputs = model(sequences, lengths)
                loss = criterion(outputs.squeeze(), labels.squeeze(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * sequences.size(0)
                train_progress.set_postfix(loss=loss.item())
            
            train_loss = total_train_loss / len(train_loader.dataset)
            val_loss = evaluate(model, val_loader, criterion, device)
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    def random_baseline(data_loader):
        total_loss = 0
        for sequences, labels, lengths in data_loader:
            total_loss += torch.abs(torch.sum(sequences, dim=1) - labels.squeeze(-1)).sum().item()
        return total_loss / len(data_loader.dataset)
    
    print(f"Random Baseline MAE: {random_baseline(val_loader)}")

    train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss (MAE): {test_loss:.4f}")

    def generate_out_of_distribution_data(start_len=1, end_len=32, samples_per_length=1000):
        data, labels = [], []
        for length in range(start_len, end_len + 1):
            for _ in range(samples_per_length):
                sequence = np.random.randint(0, 2, length)
                label = np.sum(sequence)
                data.append(sequence)
                labels.append(label)
        return data, labels

    gen_data, gen_labels = generate_out_of_distribution_data()
    lengths = list(range(1, 33))
    mae_per_length = []

    for length in lengths:
        length_data = []
        length_labels = []
        for seq, label in zip(gen_data, gen_labels):
            if len(seq) == length:
                length_data.append(seq)
                length_labels.append(label)
        
        length_dataset = BitCountingDataset(length_data, length_labels)
        length_loader = DataLoader(length_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_rnn)
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


def ocr_fun():
    def generate_word_images(word_list, image_dir, image_size=(256, 64)):
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        for word in tqdm(word_list, desc="Generating Images"):
            img = Image.new('L', image_size, color=255)
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            font_size = 24
            font = ImageFont.truetype("arial.ttf", font_size)
            bbox = draw.textbbox((0, 0), word, font=font)
            x = (image_size[0] - (bbox[2] - bbox[0])) // 2
            y = (image_size[1] - (bbox[3] - bbox[1])) // 2
            draw.text((x, y), word, font=font, fill=0)
            img.save(os.path.join(image_dir, f"{word}.png")) 

    image_dir = "./../../data/external/word_images"
    # word_list = words.words()
    # word_list = list(set(word_list))
    # word_list = word_list[:100000]
    # generate_word_images(word_list, image_dir)

    def create_image_label_lists(image_dir):
        image_paths = []
        labels = []
        for filename in os.listdir(image_dir):
            if filename.endswith(".png"):
                if ('-' in filename) or ('_' in filename):
                    continue
                label = os.path.splitext(filename)[0]
                image_paths.append(os.path.join(image_dir, filename))
                labels.append(label)
        return image_paths, labels

    image_paths, labels = create_image_label_lists(image_dir)
    max_word_length = max(len(label) for label in labels)
    print(f"Max Word Length: {max_word_length}")

    image_paths_temp, labels_temp = np.array(image_paths), np.array(labels)
    idx = np.random.permutation(len(image_paths_temp))
    image_paths, labels = image_paths_temp[idx].tolist(), labels_temp[idx].tolist()

    train_size = int(0.8 * len(image_paths))
    val_size = int(0.1 * len(image_paths))
    test_size = len(image_paths) - train_size - val_size

    train_paths, val_paths, test_paths = image_paths[:train_size], image_paths[train_size:train_size + val_size], image_paths[train_size + val_size:]
    train_labels, val_labels, test_labels = labels[:train_size], labels[train_size:train_size + val_size], labels[train_size + val_size:]

    train_dataset = OCRDataset(train_paths, train_labels, max_length=max_word_length)
    val_dataset = OCRDataset(val_paths, val_labels, max_length=max_word_length)
    test_dataset = OCRDataset(test_paths, test_labels, max_length=max_word_length)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def decode_label(one_hot_encoded):
        char_map = "@ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        decoded = ""
        for i in range(one_hot_encoded.shape[0]):
            idx = torch.argmax(one_hot_encoded[i]).item()
            if idx == 0:
                break
            decoded += char_map[idx]
        return decoded
    
    def get_weights(labels):
        weigths = torch.ones(53, dtype=torch.float)
        char_map = "@ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        weigths[0] = 0.2
        # weigths[char_map.index('h')] = 1.0
        # weigths[char_map.index('b')] = 1.0
        # weigths[char_map.index('H')] = 1.0
        return weigths

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OCRModel(num_classes=53, max_length=max_word_length).to(device)
    weigth = get_weights(train_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=weigth)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def evaluate(model, data_loader, criterion, device, print_output=False):
        model.eval()
        total_loss = 0
        correct_chars = 0
        total_chars = 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                outputs = outputs.permute(0, 2, 1)
                labels = labels.permute(0, 2, 1)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                outputs = outputs.permute(0, 2, 1)
                labels = labels.permute(0, 2, 1)
                for i in range(len(labels)):
                    predicted_label = decode_label(outputs[i])
                    true_label = decode_label(labels[i])
                    if print_output:
                        print(f"Predicted: {predicted_label}, True: {true_label}")
                    for j in range(len(true_label)):
                        if j < len(predicted_label) and predicted_label[j] == true_label[j]:
                            correct_chars += 1
                        total_chars += 1
        avg_correct_chars = correct_chars / total_chars
        total_loss = total_loss / len(data_loader)
        return total_loss, avg_correct_chars

    def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
        model.to(device)
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Training")
            
            for images, labels in train_loader_tqdm:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                outputs = outputs.permute(0, 2, 1)
                labels = labels.permute(0, 2, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss, avg_correct_chars = evaluate(model, val_loader, criterion, device)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, ANCC in Val: {avg_correct_chars:.4f}")

    def random_baseline_accuracy(labels):
        correct_chars = 0
        total_chars = 0
        for label in labels:
            predicted_label = ''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'), len(label)))
            for i in range(len(label)):
                if predicted_label[i] == label[i]:
                    correct_chars += 1
                total_chars += 1
        return correct_chars / total_chars
    
    random_accuracy = random_baseline_accuracy(val_labels)
    print(f"Random Baseline Accuracy: {random_accuracy:.4f}")

    train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)

    test_loss, avg_correct_chars = evaluate(model, test_loader, criterion, device, print_output=True)
    print(f"Test Loss: {test_loss:.4f}, ANCC in Test: {avg_correct_chars:.4f}")



# kde_fun()
# hmm_fun()
# rnn_fun()
# ocr_fun()

if __name__ == '__main__':
    while True:
        print("1. KDE")
        print("2. HMM")
        print("3. RNN")
        print("4. OCR")
        print("5. Exit")
        choice = int(input("Enter your choice: "))
        if choice == 1:
            kde_fun()
        elif choice == 2:
            hmm_fun()
        elif choice == 3:
            rnn_fun()
        elif choice == 4:
            ocr_fun()
        elif choice == 5:
            break
        else:
            print("Invalid choice. Please enter again.")