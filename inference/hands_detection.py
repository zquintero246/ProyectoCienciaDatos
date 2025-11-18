import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import time

# ============================
#  CARGA DE TU MODELO
# ============================
from train.model import BiLSTMClassifier

CKPT_PATH = "modelo_cordialidad.pth"

ckpt = torch.load(CKPT_PATH, map_location="cpu")
label_map = ckpt["label_map"]
id2label = {v:k for k,v in label_map.items()}

model = BiLSTMClassifier(
    input_dim=225,
    hidden_dim=256,
    num_layers=2,
    num_classes=len(label_map)
)

model.load_state_dict(ckpt["model_state"])
model.eval()
model.to("cpu")

# Buffer temporal para últimas ventanas
MAX_WINDOW = 45
sequence_buffer = []

# ============================
#  MODELOS MEDIAPIPE
# ============================

BaseOptions = python.BaseOptions
RunningMode = vision.RunningMode

HAND_MODEL = "models/hand_landmarker.task"
POSE_MODEL = "models/pose_landmarker_full.task"

mp_image_format = mp.ImageFormat.SRGB

# Estado global de landmarks recogidos en este frame
current_rhand = None
current_lhand = None
current_pose = None


# ======== EXTRACCIÓN DE VECTORES ========
def extract_feature_vector(r, l, p):
    def arr(obj, count):
        if obj is None:
            return np.zeros(count * 3, dtype=np.float32)

        flat = []
        for lm in obj:   # ahora sí funciona con pose
            flat += [lm.x, lm.y, lm.z]

        return np.array(flat, dtype=np.float32)

    return np.concatenate([
        arr(r, 21),
        arr(l, 21),
        arr(p, 33)
    ]).astype(np.float32)



# ============================
#  CALLBACKS MEDIAPIPE
# ============================

def hand_callback(result, output_image, timestamp_ms):
    global current_rhand, current_lhand
    if result.hand_landmarks:
        if len(result.hand_landmarks) == 1:
            # Una sola mano detectada
            # Usamos handedness del modelo
            handed = result.handedness[0][0].category_name
            if handed == "Right":
                current_rhand = result.hand_landmarks[0]
            else:
                current_lhand = result.hand_landmarks[0]

        elif len(result.hand_landmarks) == 2:
            # Dos manos
            h0 = result.handedness[0][0].category_name
            h1 = result.handedness[1][0].category_name

            if h0 == "Right":
                current_rhand = result.hand_landmarks[0]
                current_lhand = result.hand_landmarks[1]
            else:
                current_lhand = result.hand_landmarks[0]
                current_rhand = result.hand_landmarks[1]


def pose_callback(result, output_image, timestamp_ms):
    global current_pose
    if result.pose_landmarks:
        # Guarda SOLO la lista de landmarks:
        current_pose = result.pose_landmarks[0].landmark



# ============================
#    PIPE DE INFERENCIA
# ============================

def predict_from_window(seq):
    """
    seq: numpy (T,225)
    """
    with torch.no_grad():
        x = torch.from_numpy(seq).unsqueeze(0)  # (1,T,225)
        lengths = torch.tensor([seq.shape[0]])
        logits = model(x, lengths)
        pred = logits.argmax(1).item()
        return id2label[pred]


# ============================
#       MAIN
# ============================

hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL),
    running_mode=RunningMode.LIVE_STREAM,
    result_callback=hand_callback
)

pose_options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_MODEL),
    running_mode=RunningMode.LIVE_STREAM,
    result_callback=pose_callback
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

last_prediction = "..."

with vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker, \
     vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
     vision.FaceLandmarker.create_from_options(face_options) as face_landmarker:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # --- manos ---
        hands_res = hand_landmarker.detect(mp_img)
        if hands_res.hand_landmarks:
            if len(hands_res.hand_landmarks) > 0:
                current_rhand = hands_res.hand_landmarks[0]
            if len(hands_res.hand_landmarks) > 1:
                current_lhand = hands_res.hand_landmarks[1]

        # --- pose ---
        pose_res = pose_landmarker.detect(mp_img)
        if pose_res.pose_landmarks:
            current_pose = pose_res.pose_landmarks[0].landmark

        # crear feature vector
        if current_rhand and current_lhand and current_pose:
            vec = extract_feature_vector(current_rhand, current_lhand, current_pose)
            vec = torch.tensor(vec).unsqueeze(0).float()
            logits = model(vec)
            pred = torch.argmax(logits, dim=1).item()
            text = idx_to_class[pred]

            cv2.putText(frame, text, (40, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.4, (0,255,0), 3)

        cv2.imshow("LSC detector", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break


cap.release()
cv2.destroyAllWindows()