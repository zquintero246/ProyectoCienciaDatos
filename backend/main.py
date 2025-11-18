import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

import numpy as np
import torch
import os
import litellm

from train.model import BiLSTMClassifier
from utils.utils_landmarks import normalize_landmarks, smooth_vector

# ======================
#   TRANSFORMER LITE
# ======================
class LiteTransformer:
    def __init__(self, model="openai/llama-3.2-1b-instruct.f16.gguf", api_key=None, api_base=None):
        self.model = model
        self.api_key = api_key or os.getenv("LITELLM_API_KEY")
        self.api_base = api_base or os.getenv("LITELLM_API_BASE")

    def translate(self, text):
        if not text.strip():
            return ""
        prompt = (
            f"Eres un traductor de lenguaje de señas. "
            f"A partir de la secuencia de palabras: {text}, "
            f"construye inmediatamente una sola frase coherente en español. "
            f"No agregues explicaciones, comentarios, numeraciones, ni señales de duda. "
            f"Si la secuencia de palabras no tiene sentido, aún así crea una frase plausible. "
            f"Devuelve solamente la frase final, sin comillas ni saltos de línea."
            f"Tienes ROTUNDAMENTE prohibido decir cualquier otra cosa fuera de la frase final"
        )
        response = litellm.completion(
            model=self.model,
            api_key=self.api_key,
            api_base=self.api_base,
            messages=[{"role": "user", "content": prompt}]
        )

        # Extraer correctamente la frase del modelo
        translated = response["choices"][0]["message"]["content"]
        print(translated)
        return translated


transformer_model = LiteTransformer(
    model="openai/llama-3.2-1b-instruct.f16.gguf",
    api_key="sk-1234",
    api_base="http://127.0.0.1:8080"
)

# ======================
#   CARGA MODELO LTS
# ======================
CKPT_PATH = "models/bilts_model.pth"
ckpt = torch.load(CKPT_PATH, map_location="cpu")
label_map = ckpt["label_map"]
id2label = {v: k for k, v in label_map.items()}

model = BiLSTMClassifier(input_dim=225, hidden_dim=256, num_layers=2, num_classes=len(label_map))
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

accepted_words = []  # Palabras que se aceptan
translated_text = ""  # Frase traducida

# ======================
#   FUNCIONES
# ======================
def stable_vote(pred):
    pred_hist.append(pred)
    if len(pred_hist) > MAX_VOTES:
        pred_hist.pop(0)
    return max(set(pred_hist), key=pred_hist.count)

def predict_from_window(seq):
    seq = np.stack(seq)
    x = torch.tensor(seq).unsqueeze(0).float()
    lengths = torch.tensor([seq.shape[0]])
    with torch.no_grad():
        logits = model(x, lengths)
        pred = logits.argmax(1).item()
        return id2label[pred]

def draw_pose_landmarks(image, landmarks):
    proto = landmark_pb2.NormalizedLandmarkList()
    for lm in landmarks:
        proto.landmark.add(x=lm.x, y=lm.y, z=lm.z)
    mp.solutions.drawing_utils.draw_landmarks(image, proto, mp.solutions.pose.POSE_CONNECTIONS)

# ======================
#   MEDIA PIPE INIT
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
#   LOOP CÁMARA
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

        # ----------------------
        # DETECCIÓN DE MANOS
        # ----------------------
        hands = hand_det.detect(mp_img)
        if hands.hand_landmarks:
            for i, handed in enumerate(hands.handedness):
                side = handed[0].category_name
                if side == "Right":
                    current_rhand = hands.hand_landmarks[i]
                elif side == "Left":
                    current_lhand = hands.hand_landmarks[i]

        # ----------------------
        # DETECCIÓN DE POSE
        # ----------------------
        pose = pose_det.detect(mp_img)
        if pose.pose_landmarks:
            current_pose = pose.pose_landmarks[0]

        # ----------------------
        # EXTRACCIÓN DE FEATURES
        # ----------------------
        if current_pose is not None:
            vec = normalize_landmarks(current_rhand, current_lhand, current_pose)
            vec = smooth_vector(vec, prev_vec)
            if vec is not None:
                prev_vec = vec
                sequence_buffer.append(vec)
                if len(sequence_buffer) > MAX_WINDOW:
                    sequence_buffer.pop(0)

                if len(sequence_buffer) > 20:
                    raw_pred = predict_from_window(sequence_buffer)
                    last_pred = stable_vote(raw_pred)

        # ----------------------
        # DIBUJAR LANDMARKS Y TEXTO
        # ----------------------
        cv2.putText(frame, f"Predicción: {last_pred}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Aceptadas: {' '.join(accepted_words)}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Traducción: {translated_text}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        if current_pose is not None:
            draw_pose_landmarks(frame, current_pose)

        cv2.imshow("Reconocimiento LSC — Cordialidad", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC salir
            break
        elif key == ord('a'):  # Aceptar palabra
            accepted_words.append(last_pred)
        elif key == ord('t'):  # Traducir secuencia
            translated_text = transformer_model.translate(' '.join(accepted_words))

cap.release()
cv2.destroyAllWindows()
