import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class CnnAutoencoder(nn.Module):
    def __init__(self, input_channels=1, filter_sizes=[16, 32, 64], kernel_size=5):
        super(CnnAutoencoder, self).__init__()

        if (len(filter_sizes)==2):
            self.encoder = nn.Sequential(
                nn.Conv2d(input_channels, filter_sizes[0], kernel_size, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size, stride=2, padding=2)
            )

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(filter_sizes[1], filter_sizes[0], kernel_size, stride=2, padding=2, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(filter_sizes[0], input_channels, kernel_size, stride=2, padding=2, output_padding=1),
                nn.Sigmoid()
            )

        elif (len(filter_sizes)==3):
            self.encoder = nn.Sequential(
                nn.Conv2d(input_channels, filter_sizes[0], kernel_size, stride=2, padding=kernel_size//2),
                nn.ReLU(),
                nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size, stride=2, padding=kernel_size//2),
                nn.ReLU(),
                nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_size, stride=1, padding=0)
            )
            
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(filter_sizes[2], filter_sizes[1], kernel_size, stride=1, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(filter_sizes[1], filter_sizes[0], kernel_size, stride=2, padding=kernel_size//2, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(filter_sizes[0], input_channels, kernel_size, stride=2, padding=kernel_size//2, output_padding=1),
                nn.Sigmoid()
            )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def loss(self, y_pred, y_true):
        return F.mse_loss(y_pred, y_true)

    def evaluate(self, data_loader, device):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(device)
                y_pred = self.forward(x)
                loss = self.loss(y_pred, x)
                total_loss += loss.item()
        return total_loss / len(data_loader)
    
    def train_model(self, optimizer, train_loader, val_loader, num_epochs=10, device='cpu', tune=False):
        self.to(device)
        train_losses, val_losses = [], []
        for epoch in range(num_epochs):
            self.train()
            if not tune:
                train_progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
            else:
                train_progress = enumerate(train_loader)

            total_train_loss = 0
            for i, (x, _) in train_progress:
                x = x.to(device)
                optimizer.zero_grad()
                y_pred = self(x)
                loss = self.loss(y_pred, x)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            val_loss = self.evaluate(val_loader, device)
            
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            
            if not tune:
                print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        return train_losses, val_losses
