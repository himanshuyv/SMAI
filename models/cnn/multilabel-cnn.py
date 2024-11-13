import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class CNN_Multilabel(nn.Module):
    def __init__(self, num_classes=33, num_conv_layers=3, dropout_rate=0.0, task='multilabel_classification'):
        super(CNN_Multilabel, self).__init__()

        self.task = task
        self.num_classes = num_classes
        self.num_conv_layers = num_conv_layers
        self.dropout_rate = dropout_rate
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for i in range(num_conv_layers):
            if i == 0:
                setattr(self, f'conv{i+1}', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1))
            else:
                setattr(self, f'conv{i+1}', nn.Conv2d(in_channels=32 * (2 ** (i - 1)), out_channels=32 * (2 ** i), kernel_size=3, stride=2, padding=1))
            
            setattr(self, f'dropout{i+1}', nn.Dropout(dropout_rate))

        self.fc1 = None
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.fc_dropout = nn.Dropout(dropout_rate)

    def _initialize_fc(self, input_shape, device):
        dummy_input = torch.zeros(1, *input_shape).to(device)
        with torch.no_grad():
            output = self._forward_conv(dummy_input)
        flattened_size = output.view(-1).shape[0]
        self.fc1 = nn.Linear(flattened_size, 128).to(device)

    def _forward_conv(self, x):
        for i in range(self.num_conv_layers):
            conv_layer = getattr(self, f'conv{i+1}')
            dropout_layer = getattr(self, f'dropout{i+1}')
            x = self.pool(self.relu(conv_layer(x)))
            x = dropout_layer(x)
        return x

    def forward(self, x):
        if self.fc1 is None:
            self._initialize_fc(x.shape[1:], x.device)

        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_accuracy(self, y_pred, y_true):
        if self.task == 'multilabel_classification':
            y_pred_labels = self.convert_to_labels(y_pred)
            y_true_labels = self.convert_to_labels(y_true)
            incorrect = 0
            correct = 0
            for i in range(len(y_pred_labels)):
                if y_pred_labels[i] == y_true_labels[i]:
                    correct += 1
                else:
                    incorrect += 1
            return correct / (correct + incorrect)
        else:
            return F.mse_loss(y_pred, y_true)

    def loss(self, y_pred, y_true):
        if self.task == 'multilabel_classification':
            criterion = nn.CrossEntropyLoss()
            loss = 0
            for i in range(3):
                start = i * 11
                end = (i + 1) * 11
                target_idx = torch.argmax(y_true[:, start:end], dim=1)
                loss += criterion(y_pred[:, start:end], target_idx)
            return loss
        else:
            return F.mse_loss(y_pred, y_true)
        
    def get_hamming_accuracy(self, data_loader, device):
        self.eval()
        correct = 0
        incorrect = 0
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                y_pred = self.forward(x)
                y_pred_labels = self.convert_to_labels(y_pred)
                y_true_labels = self.convert_to_labels(y)
                for i in range(len(y_pred_labels)):
                    for j in range(min(len(y_pred_labels[i]), len(y_true_labels[i]))):
                        if y_pred_labels[i][j] == y_true_labels[i][j]:
                            correct += 1
                        else:
                            incorrect += 1
        return correct / (correct + incorrect)

    def convert_to_labels(self, y_pred):
        labels = []
        batch_size = y_pred.shape[0]
        for i in range(batch_size):
            label = []
            for j in range(3):
                start = j * 11
                end = (j + 1) * 11
                cur_label = torch.argmax(y_pred[i, start:end]).item()
                if cur_label != 10:
                    label.append(cur_label)
                else:
                    break
            labels.append(label)
        return labels
    
    def train_model(self, optimizer, train_loader, val_loader, num_epochs=30, device='cpu', tune=False):
        self.to(device)
        avg_train_losses = []
        avg_val_losses = []

        for epoch in range(num_epochs):
            self.train()

            if not tune:
                train_progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
            else:
                train_progress = enumerate(train_loader)
            
            total_train_loss = 0
            for i, (x, y) in train_progress:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred = self.forward(x)
                loss = self.loss(y_pred, y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

                if not tune:
                    train_progress.set_postfix({"Loss": f"{loss.item():.4f}"})

            avg_train_loss = total_train_loss / len(train_loader)
            avg_train_losses.append(avg_train_loss)
            avg_val_loss, avg_val_accuracy = self.evaluate(val_loader, device)
            avg_val_losses.append(avg_val_loss)

            if not tune:
                print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy*100:.4f}%")

        return avg_train_losses, avg_val_losses
    
    def evaluate(self, data_loader, device):
        self.eval()
        total_loss = 0.0
        total_accuracy = 0.0

        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                y_pred = self.forward(x)
                total_loss += self.loss(y_pred, y).item()
                total_accuracy += self.get_accuracy(y_pred, y)
        
        avg_loss = total_loss / len(data_loader)
        avg_accuracy = total_accuracy / len(data_loader)
        
        return avg_loss, avg_accuracy