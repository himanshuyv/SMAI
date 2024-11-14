import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class OCRModel(nn.Module):
    def __init__(self, cnn_output_dim, rnn_hidden_dim, num_classes, dropout_prob=0.5):
        super(OCRModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.flatten_dim = (256 // 8) * (64 // 8) * 128
        self.fc = nn.Linear(self.flatten_dim, cnn_output_dim)

        self.rnn = nn.LSTM(input_size=cnn_output_dim, hidden_size=rnn_hidden_dim, num_layers=2, dropout=dropout_prob, batch_first=True)
        self.output_layer = nn.Linear(rnn_hidden_dim, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
    
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(batch_size, -1)
        
        cnn_features = self.fc(cnn_features)
        cnn_features = cnn_features.unsqueeze(1).repeat(1, 10, 1)
        rnn_output, _ = self.rnn(cnn_features)
        output = self.output_layer(rnn_output)
        return output
    


class OCRDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = torch.FloatTensor(self.data[idx]).unsqueeze(0)
        label = torch.LongTensor(self.labels[idx])
        return image, label, len(label)
    
def collate_fn_ocr(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    return images, labels