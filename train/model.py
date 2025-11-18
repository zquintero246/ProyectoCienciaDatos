import torch
import torch.nn as nn

class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim=225, hidden_dim=256, num_layers=2, num_classes=32, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, lengths=None):

        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(),
                                                  batch_first=True, enforce_sorted=False)

        out, _ = self.lstm(x)

        if lengths is not None:
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # Atención temporal
        attn_w = self.attention(out)             # (B,T,1)
        attn_w = torch.softmax(attn_w, dim=1)    # pesos

        context = (out * attn_w).sum(dim=1)      # weighted sum

        logits = self.classifier(context)
        return logits
