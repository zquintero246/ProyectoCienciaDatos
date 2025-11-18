import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import dataset
from model import LSTMWithAttention
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train():
    print("Cargando metadata...")

    df = pd.read_csv("dataset/metadata.csv")
    BASE = "dataset/dataset_cordialidad"

    df["path"] = df["path"].apply(lambda p: os.path.join(BASE, p))
    df["path"] = df["path"].apply(os.path.normpath)

    # Verificar archivos faltantes
    missing = df[~df["path"].apply(os.path.exists)]
    if len(missing) > 0:
        print("Hay rutas que NO existen en el disco:")
        print(missing["path"].to_string())
        raise FileNotFoundError("Corrige las rutas en metadata.csv")

    print("Todas las rutas son válidas.")

    # Número de clases
    num_classes = df["label_id"].nunique()

    # Dataset
    train_ds = dataset.SignSequenceDataset(df)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    # Modelo
    model = LSTMWithAttention(
        input_dim=225,
        hidden_dim=256,
        num_layers=2,
        num_classes=num_classes
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0

    # ======================
    # ENTRENAMIENTO
    # ======================
    for epoch in range(40):
        model.train()
        total, correct = 0, 0

        for x, lengths, y in train_loader:
            x = x.to(DEVICE)
            lengths = lengths.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x, lengths)

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1} - Acc {acc:.4f}")

        # Guardar mejor modelo
        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state": model.state_dict(),
                "label_map": {lbl: i for i, lbl in enumerate(sorted(df['label'].unique()))}
            }, "../backend/models/lts_transformer.pth")
            print("Modelo guardado")

if __name__ == "__main__":
    train()
