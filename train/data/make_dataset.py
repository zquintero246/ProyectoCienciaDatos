import os
import json
import numpy as np

JSON_PATH = r"/train/data/datos_reducido.json"
OUTPUT_DIR = r"C:\Users\Zabdiel Julian\Downloads\Proyectos\ProyectoCienciaDatos\train\dataset"
CATEGORY = "Cordialidad"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_vec(frame):
    """Convierte un frame en un vector 1D = [r_hand(63), l_hand(63), pose(99)]."""

    def arr(obj, n):
        if obj is None:
            return np.zeros(n * 3, dtype=np.float32)
        return np.array(obj["x"] + obj["y"] + obj["z"], dtype=np.float32)

    r = arr(frame.get("r_hand"), 21)
    l = arr(frame.get("l_hand"), 21)
    p = arr(frame.get("pose"), 33)

    return np.concatenate([r, l, p], axis=0)


print("\nCargando JSON reducido...")
with open(JSON_PATH, "r") as f:
    data = json.load(f)

print("JSON reducido cargado correctamente.")

seq_counts = {}

print("Signers encontrados:", list(data.keys()))

for signer, signer_obj in data.items():

    if CATEGORY not in signer_obj:
        continue

    cat = signer_obj[CATEGORY]
    print(f"\nProcesando {signer}/{CATEGORY}")

    for palabra, vids in cat.items():

        out_dir = os.path.join(OUTPUT_DIR, palabra)
        os.makedirs(out_dir, exist_ok=True)

        seq_counts.setdefault(palabra, 0)

        for vid, reps in vids.items():
            for rep, frames in reps.items():

                frames_sorted = sorted(
                    frames.keys(),
                    key=lambda x: int(x.split("_")[1])
                )

                seq = np.stack([extract_vec(frames[f]) for f in frames_sorted])

                out_path = os.path.join(out_dir, f"seq_{seq_counts[palabra]}.npy")
                np.save(out_path, seq)
                seq_counts[palabra] += 1


print("\nPROCESO COMPLETADO")
print("Secuencias generadas por palabra:")
for palabra, n in seq_counts.items():
    print(f"  {palabra}: {n}")

print("\nDataset guardado en:", OUTPUT_DIR)
