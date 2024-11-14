import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class BitCountingDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx]).unsqueeze(-1)
        label = torch.FloatTensor([self.labels[idx]])
        length = len(sequence)
        return sequence, label, length

def collate_fn_rnn(batch):
    sequences, labels, lengths = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths)
    return sequences_padded, labels, lengths

class BitCounterRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super(BitCounterRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_input)
        out, _ = pad_packed_sequence(packed_output, batch_first=True)
        out = out[torch.arange(out.size(0)), lengths - 1]
        out = self.fc(out)
        return out