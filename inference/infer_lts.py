import argparse
import torch
import numpy as np
from model import BiLSTMClassifier

def load_label_map(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "label_map" in ckpt:
        return ckpt["label_map"]
    # fallback
    return None

def main(args):
    ckpt = torch.load(args.ckpt, map_location="cpu")
    label_map = ckpt.get("label_map")
    if label_map is None:
        print("Warning: label_map not found in checkpoint")
        label_map = {}
    id2label = {v:k for k,v in label_map.items()}

    model = BiLSTMClassifier(input_dim=225, hidden_dim=256, num_layers=2, num_classes=len(label_map))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(args.device)

    arr = np.load(args.npy)
    x = torch.from_numpy(arr.astype('float32')).unsqueeze(0).to(args.device)  # (1,T,F)
    lengths = torch.tensor([x.shape[1]])
    with torch.no_grad():
        logits = model(x, lengths.to(args.device))
        pred = logits.argmax(1).item()
        print("Pred:", pred, id2label.get(pred, "UNK"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to best.pth or final_model.pth")
    p.add_argument("--npy", required=True, help="path to sequence .npy")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    main(args)
