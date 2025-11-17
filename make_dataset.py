import json
import h5py
import numpy as np
from tqdm import tqdm

INPUT_JSON = "datos.json"
OUTPUT_HDF5 = "dataset.hdf5"
FEATURES = 225
MAX_FRAMES = 120


def extract_features(frame):
    """Extrae 225 features: right + left + pose"""

    def flatten(hand):
        if hand is None:
            return [0]*63
        return hand["x"] + hand["y"] + hand["z"]

    # manos
    r = flatten(frame.get("r_hand"))
    l = flatten(frame.get("l_hand"))

    # pose
    pose = frame.get("pose")
    if pose:
        px = pose["x"]
        py = pose["y"]
        pz = pose["z"]
    else:
        px = py = pz = [0]*33

    pose_vec = px + py + pz

    return r + l + pose_vec


def main():
    print("Leyendo JSON grande en streaming…")

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Contando secuencias…")
    total_seq = 0
    for signer in data.values():
        for category in signer.values():
            for word in category.values():
                for video in word.values():
                    total_seq += 1

    print(f"Total de secuencias: {total_seq}")

    h5 = h5py.File(OUTPUT_HDF5, "w")
    X = h5.create_dataset("X", shape=(total_seq, MAX_FRAMES, FEATURES),
                          dtype=np.float32)
    y = h5.create_dataset("y", shape=(total_seq,), dtype=np.int32)

    label_map = {}
    label_idx = 0

    seq_i = 0

    print("Procesando dataset y escribiendo HDF5…")

    for signer, signer_data in tqdm(data.items()):
        for category, cat_data in signer_data.items():
            for word, word_data in cat_data.items():

                if word not in label_map:
                    label_map[word] = label_idx
                    label_idx += 1

                for video, vid_data in word_data.items():

                    rep = vid_data.get("rep_0") or next(iter(vid_data.values()))

                    frames = list(rep.values())

                    features = []

                    for fdata in frames:
                        features.append(extract_features(fdata))

                    features = np.array(features, dtype=np.float32)

                    if len(features) >= MAX_FRAMES:
                        features = features[:MAX_FRAMES]
                    else:
                        pad = np.zeros((MAX_FRAMES - len(features), FEATURES),
                                       dtype=np.float32)
                        features = np.vstack([features, pad])

                    X[seq_i] = features
                    y[seq_i] = label_map[word]

                    seq_i += 1

    h5.close()

    with open("labels.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    print("Conversión completa.")
    print("Archivos generados:")
    print(" - dataset.hdf5")
    print(" - labels.json")


if __name__ == "__main__":
    main()
