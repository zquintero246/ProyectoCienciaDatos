import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch

from train.model import LSTMWithAttention

from utils.utils_landmarks import normalize_landmarks
from utils.utils_landmarks import smooth_vector


# ======================
#   CARGA MODELO
# ======================
CKPT_PATH = "models/lts_transformer.pth"
ckpt = torch.load(CKPT_PATH, map_location="cpu")

label_map = ckpt["label_map"]
id2label = {v: k for k, v in label_map.items()}

model = LSTMWithAttention(
    input_dim=225,
    hidden_dim=256,
    num_layers=2,
    num_classes=len(label_map)
)
model.load_state_dict(ckpt["model_state"])
model.eval()


# ======================
#   CONSTANTES
# ======================
MAX_WINDOW = 45
sequence_buffer = []
prev_vec = None

pred_hist = []
MAX_VOTES = 12

current_rhand = None
current_lhand = None
current_pose = None


# ======================
#   FUNCIÓN DE VOTACIÓN
# ======================
def stable_vote(pred):
    pred_hist.append(pred)
    if len(pred_hist) > MAX_VOTES:
        pred_hist.pop(0)
    return max(set(pred_hist), key=pred_hist.count)


# ======================
#   PREDICCIÓN DE UNA VENTANA
# ======================
def predict_from_window(seq):
    seq = np.stack(seq)      # (T,225)
    x = torch.tensor(seq).unsqueeze(0).float()
    lengths = torch.tensor([seq.shape[0]])

    with torch.no_grad():
        logits = model(x, lengths)
        pred = logits.argmax(1).item()
        return id2label[pred]


# ======================
#   MEDIAPIPE INIT
# ======================
BaseOptions = python.BaseOptions
RunningMode = vision.RunningMode

HAND_MODEL = "models/hand_landmarker.task"
POSE_MODEL = "models/pose_landmarker_full.task"

hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL),
    running_mode=RunningMode.IMAGE
)

pose_options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_MODEL),
    running_mode=RunningMode.IMAGE
)

mp_image_format = mp.ImageFormat.SRGB


# ======================
#   LOOP DE CÁMARA
# ======================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

with vision.HandLandmarker.create_from_options(hand_options) as hand_det, \
     vision.PoseLandmarker.create_from_options(pose_options) as pose_det:

    last_pred = "..."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp_image_format, data=rgb)

        # ======================
        #   DETECCIÓN DE MANOS
        # ======================
        hands = hand_det.detect(mp_img)

        if hands.hand_landmarks:
            for i, handed in enumerate(hands.handedness):
                side = handed[0].category_name  # Left / Right
                if side == "Right":
                    current_rhand = hands.hand_landmarks[i]
                elif side == "Left":
                    current_lhand = hands.hand_landmarks[i]

        # ======================
        #   DETECCIÓN DE POSE
        # ======================
        pose = pose_det.detect(mp_img)
        if pose.pose_landmarks:
            current_pose = pose.pose_landmarks[0]


        # ======================
        #   EXTRACCIÓN DE FEATURES
        # ======================
        if current_pose is not None:
            vec = normalize_landmarks(current_rhand, current_lhand, current_pose)

            vec = smooth_vector(vec, prev_vec)
            prev_vec = vec

            sequence_buffer.append(vec)

            # Limitar tamaño buffer
            if len(sequence_buffer) > MAX_WINDOW:
                sequence_buffer.pop(0)

            # Predicción si hay suficientes frames
            if len(sequence_buffer) > 20:
                raw_pred = predict_from_window(sequence_buffer)
                last_pred = stable_vote(raw_pred)

        # ======================
        #   DIBUJAR RESULTADO
        # ======================
        cv2.putText(frame, last_pred, (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (0, 255, 0), 3)

        cv2.imshow("Reconocimiento LSC — Cordialidad", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
