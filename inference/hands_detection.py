import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import torch


from train.model import BiLSTMClassifier

CKPT_PATH = "modelo_cordialidad.pth"

ckpt = torch.load(CKPT_PATH, map_location="cpu")
label_map = ckpt["label_map"]
id2label = {v: k for k, v in label_map.items()}

model = BiLSTMClassifier(
    input_dim=225,
    hidden_dim=256,
    num_layers=2,
    num_classes=len(label_map)
)

print("Configuración del modelo:")
print("  Hidden dim:", model.hidden_dim)
print("  Num layers:", model.num_layers)

model.load_state_dict(ckpt["model_state"])
model.eval()

MAX_WINDOW = 45
sequence_buffer = []

BaseOptions = python.BaseOptions
RunningMode = vision.RunningMode

HAND_MODEL = "models/hand_landmarker.task"
POSE_MODEL = "models/pose_landmarker_full.task"

mp_image_format = mp.ImageFormat.SRGB

current_rhand = None
current_lhand = None
current_pose = None


def extract_feature_vector(r, l, p):
    def arr(obj, count):
        if obj is None:
            return np.zeros(count * 3, dtype=np.float32)

        flat = []
        for lm in obj:
            flat += [lm.x, lm.y, lm.z]
        return np.array(flat, dtype=np.float32)

    return np.concatenate([
        arr(r, 21),
        arr(l, 21),
        arr(p, 33)
    ]).astype(np.float32)



def predict_from_window(seq):
    seq = np.stack(seq)  # (T,225)

    with torch.no_grad():
        x = torch.tensor(seq).unsqueeze(0).float()  # (1,T,225)
        lengths = torch.tensor([seq.shape[0]])
        logits = model(x, lengths)
        pred = logits.argmax(1).item()
        pred_label = id2label[pred]
        return pred_label



hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL),
    running_mode=RunningMode.IMAGE
)

pose_options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_MODEL),
    running_mode=RunningMode.IMAGE
)

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
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # -------- MANOS ----------
        hands = hand_det.detect(mp_img)
        if hands.hand_landmarks:
            if hands.handedness:
                for i, handed in enumerate(hands.handedness):
                    label = handed[0].category_name  # "Left" o "Right"
                    if label == "Right":
                        current_rhand = hands.hand_landmarks[i]
                    elif label == "Left":
                        current_lhand = hands.hand_landmarks[i]

        # -------- POSE ----------
        pose = pose_det.detect(mp_img)
        if pose.pose_landmarks:
            current_pose = pose.pose_landmarks[0]


        # -------- FEATURE VECTOR ----------
        if current_pose and (current_rhand or current_lhand):
            vec = extract_feature_vector(current_rhand, current_lhand, current_pose)
            sequence_buffer.append(vec)

            if len(sequence_buffer) > MAX_WINDOW:
                sequence_buffer.pop(0)

            if len(sequence_buffer) > 20:
                last_pred = predict_from_window(sequence_buffer)

        # -------- VISUAL ----------
        cv2.putText(frame, last_pred, (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

        cv2.imshow("Reconocimiento LSC — Cordialidad", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()