import numpy as np

def normalize_landmarks(hand_r, hand_l, pose):
    """
    Normaliza:
    - centra todos los landmarks en pose[0] (pelvis)
    - escala con la distancia entre hombros
    - NO devuelve ceros cuando falta una mano -> usa None protection
    """

    def lm_list_to_np(lmlist, count):
        if lmlist is None:
            return np.zeros((count, 3), dtype=np.float32)
        arr = np.array([[lm.x, lm.y, lm.z] for lm in lmlist], dtype=np.float32)
        return arr

    hand_r = lm_list_to_np(hand_r, 21)
    hand_l = lm_list_to_np(hand_l, 21)
    pose   = lm_list_to_np(pose, 33)

    # --- Si no hay pose, no podemos normalizar ---
    if np.all(pose == 0):
        return np.zeros(225, dtype=np.float32)

    # 1) Centrar en pelvis
    origin = pose[0].copy()
    hand_r -= origin
    hand_l -= origin
    pose   -= origin

    # 2) Escalar por distancia entre hombros
    left_shoulder  = pose[11]
    right_shoulder = pose[12]
    scale = np.linalg.norm(left_shoulder - right_shoulder)
    if scale < 1e-4: scale = 1.0

    hand_r /= scale
    hand_l /= scale
    pose   /= scale

    # Concatenar
    out = np.concatenate([hand_r.flatten(),
                          hand_l.flatten(),
                          pose.flatten()])

    return out.astype(np.float32)

def smooth_vector(new_vec, prev_vec, alpha=0.7):
    if prev_vec is None:
        return new_vec
    return alpha * prev_vec + (1 - alpha) * new_vec
