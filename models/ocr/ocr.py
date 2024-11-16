import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class OCRDataset(Dataset):
    def __init__(self, image_paths, labels, max_length=20, num_classes=53):
        self.image_paths = image_paths
        self.labels = labels
        self.max_length = max_length
        self.num_classes = num_classes
        self.char2idx = self.create_char_map()
    
    def create_char_map(self):
        chars = '@ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        char_map = {}
        for idx, char in enumerate(chars):
            char_map[char] = idx
        return char_map
    
    def encode_label(self, label):
        one_hot_encoded = torch.zeros((self.max_length, self.num_classes), dtype=torch.float)
        cur_label = label
        cur_len = len(label)
        cur_label = cur_label + '@' * (self.max_length - cur_len)
        for idx, char in enumerate(cur_label[:self.max_length]):
            one_hot_encoded[idx][self.char2idx[char]] = 1.0
        return one_hot_encoded

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')
        image = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0) / 255.0
        label = self.encode_label(self.labels[idx])
        return image, label
    

class OCRModel(nn.Module):
    def __init__(self, num_classes=53, max_length=20, hidden_dim=256, num_layers=2, dropout=0.2):
        super(OCRModel, self).__init__()
        self.num_classes = num_classes
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(256 * 16 * 4 , hidden_dim)
        self.rnn = nn.RNN(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc(x).unsqueeze(1).repeat(1, self.max_length, 1)
        rnn_out, _ = self.rnn(x)
        rnn_out = self.layernorm(rnn_out)
        out = self.fc_out(rnn_out)
        return out
