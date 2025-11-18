import torch
from torch.utils.data import Dataset
import numpy as np
import os

class SignSequenceDataset(Dataset):
    def __init__(self, df, max_len=60):
        self.df = df
        self.max_len = max_len

    def pad_or_crop(self, seq):
        L = seq.shape[0]
        if L >= self.max_len:
            return seq[:self.max_len], self.max_len
        else:
            pad = np.zeros((self.max_len - L, seq.shape[1]), dtype=np.float32)
            return np.concatenate([seq, pad], axis=0), L

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['path']
        label_id = int(row['label_id'])

        seq = np.load(path).astype(np.float32)

        seq, L = self.pad_or_crop(seq)

        return torch.tensor(seq), torch.tensor(L), torch.tensor(label_id)
