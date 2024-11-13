import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class CNN(nn.Module):
    def __init__(self, num_classes=4, num_conv_layers=3, dropout_rate=0.2, task='classification'):
        super(CNN, self).__init__()
        
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
                setattr(self, f'conv{i+1}', nn.Conv2d(in_channels=32*(2**(i-1)), out_channels=32*(2**i), kernel_size=3, stride=2, padding=1))
            
            setattr(self, f'dropout{i+1}', nn.Dropout(dropout_rate))

        self.fc1 = None
        self.fc2 = nn.Linear(128, 64)
        self.fc_dropout = nn.Dropout(dropout_rate)
        
        if task == 'classification':
            self.fc3 = nn.Linear(64, num_classes)
        elif task == 'regression':
            self.fc3 = nn.Linear(64, 1)

    def _initialize_fc(self, input_shape, device):
        dummy_input = torch.zeros(1, *input_shape).to(device)
        with torch.no_grad():
            output, _ = self._forward_conv(dummy_input)
        flattened_size = output.view(-1).shape[0]
        self.fc1 = nn.Linear(flattened_size, 128).to(device)

    def _forward_conv(self, x):
        feature_maps = {}
        for i in range(self.num_conv_layers):
            conv_layer = getattr(self, f'conv{i+1}')
            dropout_layer = getattr(self, f'dropout{i+1}')
            x = self.pool(self.relu(conv_layer(x)))
            x = dropout_layer(x)
            feature_maps[f'conv{i+1}'] = x
        return x, feature_maps

    def forward(self, x):
        if self.fc1 is None:
            self._initialize_fc(x.shape[1:], x.device)

        x, feature_maps = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x, feature_maps

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            y_pred, feature_maps = self.forward(x)
        return y_pred, feature_maps

    def get_accuracy(self, y_pred, y_true):
        if self.task == 'classification':
            y_pred = torch.argmax(y_pred, dim=1)
            return (y_pred == y_true).float().mean()
        elif self.task == 'regression':
            y_pred = torch.round(y_pred)
            return (y_pred == y_true).float().mean()

    def loss(self, y_pred, y_true):
        if self.task == 'classification':
            return F.cross_entropy(y_pred, y_true)
        elif self.task == 'regression':
            return F.mse_loss(y_pred, y_true)

    def train_model(self, optimizer, train_loader, val_loader, num_epochs=10, device='cpu', tune=False):
        self.to(device)
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            self.train()
            total_train_loss = 0.0
            if not tune:
                train_progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
            else:
                train_progress = enumerate(train_loader)
            
            for i, (x, y) in train_progress:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred, _ = self.forward(x)
                loss = self.loss(y_pred, y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                
                if not tune:
                    train_progress.set_postfix({"Loss": f"{loss.item():.4f}"})
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            avg_val_loss, avg_val_accuracy = self.evaluate(val_loader, device)
            val_losses.append(avg_val_loss)
            
            if not tune:
                print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy*100:.4f}")

        return train_losses, val_losses

    def evaluate(self, data_loader, device):
        self.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                y_pred, _ = self.forward(x)
                total_loss += self.loss(y_pred, y).item()
                total_accuracy += self.get_accuracy(y_pred, y).item()
        
        avg_loss = total_loss / len(data_loader)
        avg_accuracy = total_accuracy / len(data_loader)
        
        return avg_loss, avg_accuracy
    

    def visulize_feature_maps(self, train_loader, device='cpu'):
        self.eval()
        with torch.no_grad():
            fig, ax = plt.subplots(6, 3)
            cnt = 0
            for x, y in train_loader:
                x = x.to(device)
                if cnt == 6:
                    break
                y_pred, feature_maps = self.forward(x)
                if (y_pred.argmax(dim=1) == y).all():
                    cnt += 1
                else:
                    continue
                for i in range(3):
                    ax[cnt-1, i].imshow(feature_maps[f'conv{i+1}'][0, 0].cpu().numpy(), cmap='gray')
                    ax[cnt-1, i].set_title(f'Layer{i+1} ')
                    ax[cnt-1, i].axis('off')
            plt.tight_layout()
            plt.show()
