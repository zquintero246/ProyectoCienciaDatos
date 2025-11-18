import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=225, hidden_dim=256, num_layers=2, num_classes=32, dropout=0.3, bidirectional=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=dropout)
        factor = 2 if bidirectional else 1
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * factor, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, lengths=None):
        packed = None
        if lengths is not None:
            # pack
            lengths_cpu = lengths.cpu()
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
            out_packed, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        else:
            out, _ = self.lstm(x)
        if lengths is not None:
            mask = (torch.arange(out.size(1), device=out.device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(2)
            summed = (out * mask).sum(dim=1)
            meaned = summed / lengths.unsqueeze(1).to(out.dtype)
        else:
            meaned = out.mean(dim=1)
        logits = self.classifier(meaned)
        return logits
