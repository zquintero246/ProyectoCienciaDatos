import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st
import cv2
import numpy as np
import torch
import os
from train.model import BiLSTMClassifier
from utils.utils_landmarks import normalize_landmarks, smooth_vector
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import litellm

# ======================
#   MODELO DE TRADUCCIÓN
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
            f"No agregues explicaciones ni comentarios. "
            f"Devuelve solamente la frase final."
        )
        response = litellm.completion(
            model=self.model,
            api_key=self.api_key,
            api_base=self.api_base,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]

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

mp_image_format = python.ImageFormat.SRGB

# ======================
#   FUNCIONES AUX
# ======================
sequence_buffer = []
prev_vec = None
pred_hist = []
MAX_WINDOW = 45
MAX_VOTES = 12

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

# ======================
#   STREAMLIT INTERFAZ
# ======================
st.title("Reconocimiento de Lengua de Señas con Traducción")

run_camera = st.checkbox("Activar cámara")
accepted_words = st.session_state.get("accepted_words", [])
translated_text = st.session_state.get("translated_text", "")

if run_camera:
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    with vision.HandLandmarker.create_from_options(hand_options) as hand_det, \
         vision.PoseLandmarker.create_from_options(pose_options) as pose_det:

        last_pred = "..."
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = python.Image(image_format=mp_image_format, data=rgb)

            # Detección de manos
            hands = hand_det.detect(mp_img)
            current_rhand = current_lhand = None
            if hands.hand_landmarks:
                for i, handed in enumerate(hands.handedness):
                    side = handed[0].category_name
                    if side == "Right":
                        current_rhand = hands.hand_landmarks[i]
                    elif side == "Left":
                        current_lhand = hands.hand_landmarks[i]

            # Detección de pose
            pose = pose_det.detect(mp_img)
            current_pose = pose.pose_landmarks[0] if pose.pose_landmarks else None

            # Extracción de features
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

            # Mostrar info en frame
            cv2.putText(frame, f"Predicción: {last_pred}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Aceptadas: {' '.join(accepted_words)}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Traducción: {translated_text}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if st.button("Aceptar palabra"):
                accepted_words.append(last_pred)
                st.session_state["accepted_words"] = accepted_words

            if st.button("Traducir secuencia"):
                translated_text = transformer_model.translate(' '.join(accepted_words))
                st.session_state["translated_text"] = translated_text

    cap.release()
