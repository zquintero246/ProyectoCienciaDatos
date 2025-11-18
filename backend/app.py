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
import mediapipe as mp
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
        try:
            response = litellm.completion(
                model=self.model,
                api_key=self.api_key,
                api_base=self.api_base,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            st.error(f"Error en traducción: {e}")
            return text  # Retorna el texto original si hay error


transformer_model = LiteTransformer(
    model="openai/llama-3.2-1b-instruct.f16.gguf",
    api_key="sk-1234",
    api_base="http://127.0.0.1:8080"
)

# ======================
#   CARGA MODELO LTSM
# ======================
CKPT_PATH = "models/bilts_model.pth"
if not os.path.exists(CKPT_PATH):
    st.error(f"Modelo no encontrado en: {CKPT_PATH}")
    st.stop()

try:
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    label_map = ckpt["label_map"]
    id2label = {v: k for k, v in label_map.items()}

    model = BiLSTMClassifier(input_dim=225, hidden_dim=256, num_layers=2, num_classes=len(label_map))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
except Exception as e:
    st.error(f"Error cargando el modelo: {e}")
    st.stop()

# ======================
#   MEDIA PIPE INIT
# ======================
BaseOptions = python.BaseOptions
RunningMode = vision.RunningMode

HAND_MODEL = "models/hand_landmarker.task"
POSE_MODEL = "models/pose_landmarker_full.task"

# Verificar que los modelos existen
if not os.path.exists(HAND_MODEL):
    st.error(f"Modelo de manos no encontrado en: {HAND_MODEL}")
if not os.path.exists(POSE_MODEL):
    st.error(f"Modelo de pose no encontrado en: {POSE_MODEL}")

try:
    hand_options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=RunningMode.IMAGE,
        num_hands=2
    )

    pose_options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=RunningMode.IMAGE
    )
except Exception as e:
    st.error(f"Error inicializando MediaPipe: {e}")
    st.stop()

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
    if len(pred_hist) == 0:
        return "..."
    return max(set(pred_hist), key=pred_hist.count)


def predict_from_window(seq):
    if len(seq) == 0:
        return "..."
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

# Inicializar variables de sesión
if "accepted_words" not in st.session_state:
    st.session_state.accepted_words = []
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""
if "last_pred" not in st.session_state:
    st.session_state.last_pred = "..."
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

accepted_words = st.session_state.accepted_words
translated_text = st.session_state.translated_text

# Controles fuera del bucle de cámara
col1, col2, col3 = st.columns(3)
with col1:
    accept_btn = st.button("Aceptar palabra", key="accept_btn")
with col2:
    translate_btn = st.button("Traducir secuencia", key="translate_btn")
with col3:
    clear_btn = st.button("Limpiar todo", key="clear_btn")

if accept_btn and st.session_state.last_pred != "...":
    accepted_words.append(st.session_state.last_pred)
    st.session_state.accepted_words = accepted_words
    st.rerun()

if translate_btn and accepted_words:
    try:
        translated_text = transformer_model.translate(' '.join(accepted_words))
        st.session_state.translated_text = translated_text
        st.rerun()
    except Exception as e:
        st.error(f"Error en traducción: {e}")

if clear_btn:
    st.session_state.accepted_words = []
    st.session_state.translated_text = ""
    st.session_state.last_pred = "..."
    st.rerun()

# Mostrar estado actual
st.subheader("Palabras aceptadas:")
st.write(" ".join(accepted_words) if accepted_words else "Ninguna")
st.subheader("Traducción:")
st.write(translated_text if translated_text else "No traducido")

# Control de cámara
run_camera = st.checkbox("Activar cámara", value=st.session_state.camera_active)
st.session_state.camera_active = run_camera

if run_camera:
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("No se pudo acceder a la cámara")
        st.stop()

    try:
        with vision.HandLandmarker.create_from_options(hand_options) as hand_det, \
                vision.PoseLandmarker.create_from_options(pose_options) as pose_det:

            while cap.isOpened() and st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Error leyendo frame de la cámara")
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                # Detección de manos
                try:
                    hand_result = hand_det.detect(mp_image)
                    current_rhand = current_lhand = None

                    if hand_result.hand_landmarks:
                        for i, handedness in enumerate(hand_result.handedness):
                            if handedness and len(handedness) > 0:
                                side = handedness[0].category_name
                                if side == "Right":
                                    current_rhand = hand_result.hand_landmarks[i]
                                elif side == "Left":
                                    current_lhand = hand_result.hand_landmarks[i]
                except Exception as e:
                    st.warning(f"Error en detección de manos: {e}")

                # Detección de pose
                try:
                    pose_result = pose_det.detect(mp_image)
                    current_pose = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
                except Exception as e:
                    st.warning(f"Error en detección de pose: {e}")
                    current_pose = None

                # Extracción de features
                try:
                    if current_pose is not None:
                        vec = normalize_landmarks(current_rhand, current_lhand, current_pose)
                        if vec is not None:
                            vec = smooth_vector(vec, prev_vec)
                            prev_vec = vec
                            sequence_buffer.append(vec)
                            if len(sequence_buffer) > MAX_WINDOW:
                                sequence_buffer.pop(0)

                            if len(sequence_buffer) > 20:
                                raw_pred = predict_from_window(sequence_buffer)
                                st.session_state.last_pred = stable_vote(raw_pred)
                except Exception as e:
                    st.warning(f"Error procesando landmarks: {e}")

                # Mostrar info en frame
                frame_display = frame.copy()
                cv2.putText(frame_display, f"Prediccion: {st.session_state.last_pred}",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame_display, f"Aceptadas: {' '.join(accepted_words)}",
                            (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame_display, f"Traduccion: {translated_text}",
                            (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

                FRAME_WINDOW.image(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB))

    except Exception as e:
        st.error(f"Error en el procesamiento: {e}")
    finally:
        cap.release()
else:
    st.info("Cámara desactivada - active la cámara para comenzar el reconocimiento")