import os, csv
import numpy as np

DATA_DIR = "/home/user/ProyectoCienciaDatos/train/data/dataset_cordialidad"
OUT_CSV = "/home/user/ProyectoCienciaDatos/train/data/metadata.csv"

rows = []
labels = sorted(os.listdir(DATA_DIR))
label_map = {label: idx for idx, label in enumerate(labels)}

for label in labels:
    lab_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(lab_dir): continue
    for fn in os.listdir(lab_dir):
        if not fn.endswith(".npy"): continue
        path = os.path.join(lab_dir, fn)
        try:
            arr = np.load(path, mmap_mode='r')
            n_frames = arr.shape[0]
            rows.append((path, label, label_map[label], n_frames))
        except Exception as e:
            print("Error loading", path, e)

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["path","label","label_id","n_frames"])
    writer.writerows(rows)

print("Wrote", len(rows), "rows to", OUT_CSV)
print("Label map:", label_map)
