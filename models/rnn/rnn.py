import torch
import torch.nn as nn
from torch.utils.data import Dataset

class RNNDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        self.max_length = max(len(seq) for seq in sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        length = len(sequence)
        padded_sequence = torch.zeros((self.max_length, 1), dtype=torch.float32)
        padded_sequence[:length, 0] = torch.FloatTensor(sequence)

        label = torch.FloatTensor([label])
        return padded_sequence, label, length


class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        out, _ = self.rnn(x)
        batch_size = x.size(0)
        last_outputs = torch.zeros(batch_size, out.size(2), device=out.device)
        for i, length in enumerate(lengths):
            last_outputs[i] = out[i, length - 1]
        out = self.fc(last_outputs)
        return out
