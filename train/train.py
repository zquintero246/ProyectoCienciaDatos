import os
import math
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SignDataset, collate_fn
from model import BiLSTMClassifier
from tqdm import tqdm

# CONFIG
METADATA = "/home/user/ProyectoCienciaDatos/train/data/metadata.csv"
OUT_DIR = "/home/user/ProyectoCienciaDatos/train/checkpoints"
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 30
BATCH_SIZE = 48   # adjust if OOM; V100 16GB should handle 32-64 depending on length
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
PATIENCE = 6      # early stopping patience on val loss
USE_AMP = True

# prepare datasets
train_ds = SignDataset(METADATA, split="train")
val_ds = SignDataset(METADATA, split="val")
test_ds = SignDataset(METADATA, split="test")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

num_classes = len(train_ds.label2id)
input_dim = 225  # as built earlier

model = BiLSTMClassifier(input_dim=input_dim, hidden_dim=256, num_layers=2, num_classes=num_classes, dropout=0.3).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
criterion = nn.CrossEntropyLoss()

scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

best_val = 1e9
stale = 0

def eval_epoch(loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for x, lengths, mask, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            lengths = lengths.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = model(x, lengths)
                loss = criterion(logits, y)
            preds = logits.argmax(1)
            total += y.size(0)
            total_correct += (preds == y).sum().item()
            total_loss += loss.item() * y.size(0)
    return total_loss/total, total_correct/total

for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
    for x, lengths, mask, y in pbar:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        lengths = lengths.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(x, lengths)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * y.size(0)
        pbar.set_postfix(loss=running_loss / ((pbar.n+1) * BATCH_SIZE))
    train_loss = running_loss / len(train_ds)

    val_loss, val_acc = eval_epoch(val_loader)
    print(f"\nEpoch {epoch} TrainLoss={train_loss:.4f} ValLoss={val_loss:.4f} ValAcc={val_acc:.4f}")

    scheduler.step(val_loss)

    # checkpoint best
    if val_loss < best_val:
        best_val = val_loss
        stale = 0
        ckpt = {
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "label_map": train_ds.label2id
        }
        torch.save(ckpt, os.path.join(OUT_DIR, "best.pth"))
        print("Saved best.pth")
    else:
        stale += 1
        print(f"No improvement ({stale}/{PATIENCE})")
        if stale >= PATIENCE:
            print("Early stopping triggered.")
            break

# final test
print("Evaluating on test set with best checkpoint...")
ckpt = torch.load(os.path.join(OUT_DIR, "best.pth"), map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=2, pin_memory=True)
test_loss, test_acc = eval_epoch(test_loader)
print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

# save final simple model for inference
torch.save({"model_state": model.state_dict(), "label_map": train_ds.label2id}, os.path.join(OUT_DIR, "final_model.pth"))
print("Saved final_model.pth")
