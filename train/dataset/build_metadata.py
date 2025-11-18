import os
import csv
import json
import numpy as np

# =============================================================
# CONFIGURACIÓN
# =============================================================
# Punto donde está tu dataset:
DATASET_DIR = os.path.normpath(r"/train/dataset/dataset_cordialidad")

# Donde guardar metadata:
OUT_CSV = "data/metadata.csv"
OUT_LABELMAP = "data/label_map.json"

# Extensión a buscar
EXT = ".npy"

# =============================================================
# CREACIÓN DE METADATA
# =============================================================
def generate_metadata():
    print("📂 Escaneando dataset en:", DATASET_DIR)

    if not os.path.isdir(DATASET_DIR):
        raise RuntimeError("❌ ERROR: El directorio del dataset no existe.")

    rows = []
    bad_files = []

    # Detectar clases = nombres de subcarpetas
    labels = sorted([
        name for name in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, name))
    ])

    print("\n🔎 Clases detectadas:")
    for lbl in labels:
        print("  •", lbl)

    label_map = {label: idx for idx, label in enumerate(labels)}

    print("\n📝 Mapeo label → id:")
    for k, v in label_map.items():
        print(f"  {k}: {v}")

    # ---------------------------------------------------------
    # Recorrer dataset
    # ---------------------------------------------------------
    for label in labels:
        class_dir = os.path.join(DATASET_DIR, label)

        for fname in os.listdir(class_dir):
            if not fname.endswith(EXT):
                continue

            full_path = os.path.join(class_dir, fname)
            rel_path = os.path.relpath(full_path, DATASET_DIR)

            try:
                arr = np.load(full_path, mmap_mode="r")
                n_frames = arr.shape[0]

                rows.append([rel_path, label, label_map[label], n_frames])

            except Exception as e:
                print(f"❌ Archivo dañado: {full_path} ({e})")
                bad_files.append(full_path)

    # =============================================================
    # GUARDAR CSV
    # =============================================================
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "label_id", "n_frames"])
        writer.writerows(rows)

    print(f"\n📄 Metadata generada: {OUT_CSV}")
    print(f"   Total secuencias: {len(rows)}")

    # =============================================================
    # GUARDAR LABEL MAP
    # =============================================================
    with open(OUT_LABELMAP, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=4, ensure_ascii=False)

    print(f"📘 Label map guardado en: {OUT_LABELMAP}")

    # =============================================================
    # Reporte final
    # =============================================================
    if bad_files:
        print("\n⚠ Archivos dañados (no se incluyeron):")
        for bf in bad_files:
            print("   -", bf)
    else:
        print("\n✔ No se encontraron archivos dañados.")

    print("\n🎉 Listo.")

# =============================================================
# RUN
# =============================================================
if __name__ == "__main__":
    generate_metadata()
