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
from models.ocr.ocr import OCRModel, OCRDataset, collate_fn_ocr

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
                loss = criterion(outputs.squeeze(), labels.squeeze(-1))
                total_loss += loss.item() * sequences.size(0)
        return total_loss / len(data_loader.dataset)

    data, labels = generate_bit_count_data()
    train_data, temp_data, train_labels, temp_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
    val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=42)

    train_dataset = BitCountingDataset(train_data, train_labels)
    val_dataset = BitCountingDataset(val_data, val_labels)
    test_dataset = BitCountingDataset(test_data, test_labels)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_rnn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_rnn)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_rnn)

    model = BitCounterRNN(input_size=1, hidden_size=32, num_layers=1)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_epochs = 10

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


    def generate_out_of_distribution_data(start_len=1, end_len=32, samples_per_length=1000):
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
    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_rnn)

    lengths = list(range(1, 33))
    mae_per_length = []

    for length in lengths:
        length_data = [seq for seq in ood_data if len(seq) == length]
        length_labels = [label for seq, label in zip(ood_data, ood_labels) if len(seq) == length]
        
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

def rnn2_fun():
    def generate_word_images(word_list, image_dir, image_size=(256, 64)):
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        
        for word in tqdm(word_list, desc="Generating Images"):
            img = Image.new('L', image_size, color=255)
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), word, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = (image_size[0] - text_width) // 2
            text_y = (image_size[1] - text_height) // 2
            draw.text((text_x, text_y), word, font=font, fill=0)
            img.save(os.path.join(image_dir, f"{word}.png")) 

    image_dir = "word_images"
    word_list = words.words()[:10]
    generate_word_images(word_list, image_dir)
    def average_correct_characters(preds, labels):
        total_correct = 0
        total_chars = 0
        
        for pred, label in zip(preds, labels):
            pred_chars = torch.argmax(pred, dim=1)
            correct = (pred_chars == label).sum().item()
            total_correct += correct
            total_chars += len(label)
        
        return total_correct / total_chars

    def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, device='cuda'):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        model.to(device)
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            
            for images, labels, lengths in train_loader:
                images, labels, lengths = images.to(device), labels.to(device), lengths.to(device)
                
                optimizer.zero_grad()
                outputs, _ = model(images, lengths)
                
                outputs = outputs.view(-1, outputs.size(-1))
                labels = labels.view(-1)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            model.eval()
            total_val_loss = 0
            total_val_correct_chars = 0
            
            with torch.no_grad():
                for images, labels, lengths in val_loader:
                    images, labels, lengths = images.to(device), labels.to(device), lengths.to(device)
                    outputs, _ = model(images, lengths)

                    outputs_flat = outputs.view(-1, outputs.size(-1))
                    labels_flat = labels.view(-1)
                    val_loss = criterion(outputs_flat, labels_flat)
                    total_val_loss += val_loss.item()
                    
                    total_val_correct_chars += average_correct_characters(outputs, labels)
            
            avg_val_loss = total_val_loss / len(val_loader)
            avg_val_accuracy = total_val_correct_chars / len(val_loader)
            
            val_losses.append(avg_val_loss)
            val_accuracies.append(avg_val_accuracy)
            
            print(f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Validation Loss: {avg_val_loss:.4f}, "
                f"Validation Accuracy (Correct Characters): {avg_val_accuracy:.4f}")
        
        return train_losses, val_losses, val_accuracies
    
    def load_image(image_path, image_size=(256, 64)):
        img = Image.open(image_path).convert('L')
        img = img.resize(image_size)
        img = np.array(img) / 255.0
        img = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)
        return img
    
    def load_images(image_dir):
        images = []
        labels = []
        for image_path in os.listdir(image_dir):
            image = load_image(os.path.join(image_dir, image_path))
            images.append(image)
            label = image_path.split('.')[0]
        return images, labels
    
    data, labels = load_images(image_dir)
    train_dataset = OCRDataset(data, labels)
    val_dataset = OCRDataset(data, labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_ocr)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn_ocr)

    model = OCRModel(cnn_output_dim=128, rnn_hidden_dim=32, num_classes=100)
    train_losses, val_losses, val_accuracies = train_model(model, train_loader, val_loader)

    print(train_losses, val_losses, val_accuracies)




# kde_fun()
# hmm_fun()
# rnn_fun()
rnn2_fun()

# if __name__ == '__main__':
#     while True:
#         print("1. KDE")
#         print("2. HMM")
#         print("3. RNN")
#         print("4. Exit")
#         choice = int(input("Enter your choice: "))
#         if choice == 1:
#             kde_fun()
#         elif choice == 2:
#             hmm_fun()
#         elif choice == 3:
#             rnn_fun()
#         elif choice == 4:
#             break
#         else:
#             print("Invalid choice. Please enter again.")