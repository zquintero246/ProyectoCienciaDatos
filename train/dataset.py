import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple

class SignDataset(Dataset):
    def __init__(self, metadata_csv: str, split: str = "train", splits=(0.8,0.1,0.1), seed=42):
        df = pd.read_csv(metadata_csv)
        # basic stratified split by label
        labels = df['label'].unique().tolist()
        train_rows, val_rows, test_rows = [], [], []
        rng = np.random.RandomState(seed)
        for lab in labels:
            sub = df[df['label']==lab].sample(frac=1, random_state=seed).reset_index(drop=True)
            n = len(sub)
            n_val = max(1, int(n * splits[1]))
            n_test = max(1, int(n * splits[2]))
            n_train = n - n_val - n_test
            if n_train < 1:
                n_train = max(1, n - n_val - n_test)
            train_rows.append(sub.iloc[:n_train])
            val_rows.append(sub.iloc[n_train:n_train+n_val])
            test_rows.append(sub.iloc[n_train+n_val:n_train+n_val+n_test])
        train_df = pd.concat(train_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
        val_df = pd.concat(val_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
        test_df = pd.concat(test_rows).sample(frac=1, random_state=seed).reset_index(drop=True)

        if split == "train":
            self.df = train_df
        elif split == "val":
            self.df = val_df
        elif split == "test":
            self.df = test_df
        else:
            raise ValueError("split must be train/val/test")

        self.label2id = {l:i for i,l in enumerate(sorted(df['label'].unique().tolist()))}
        self.id2label = {v:k for k,v in self.label2id.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        arr = np.load(row["path"]).astype(np.float32)  # (T, feat)
        tensor = torch.from_numpy(arr)  # (T,feat)
        label_id = int(row["label_id"])
        return tensor, label_id

def collate_fn(batch: List[Tuple[torch.Tensor, int]], max_len:int=None):
    # batch: list of (T_i, feat), label
    seqs, labels = zip(*batch)
    lengths = [s.shape[0] for s in seqs]
    feat = seqs[0].shape[1]
    if max_len is None:
        max_len = max(lengths)
    batch_size = len(seqs)
    out = torch.zeros(batch_size, max_len, feat, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for i, s in enumerate(seqs):
        L = min(s.shape[0], max_len)
        out[i, :L] = s[:L]
        mask[i, :L] = 1
    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor([min(l, max_len) for l in lengths], dtype=torch.long)
    return out, lengths, mask, labels
