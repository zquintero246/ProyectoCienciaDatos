import cv2
import mediapipe as mp
import pygame
import time
import os
import numpy as np

# Inicializar pygame para audio
pygame.mixer.init()
pygame.mixer.set_num_channels(20)  # Más canales para ambas manos

# Screen settings
screen_width = 1280
screen_height = 720
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

# Configuración de notas para 10 DEDOS (ambas manos)
NOTES_CONFIG = {
    # MANO IZQUIERDA (dedos más graves)
    "menique_izq": {
        "landmark": 20,  # PINKY_TIP (mano izquierda)
        "note": "C4",
        "file": "sounds/C4.mp3",
        "color": (128, 0, 128),  # Púrpura
        "hand": "left"
    },
    "anular_izq": {
        "landmark": 16,  # RING_FINGER_TIP
        "note": "D4",
        "file": "sounds/D4.mp3",
        "color": (0, 255, 255),  # Cyan
        "hand": "left"
    },
    "medio_izq": {
        "landmark": 12,  # MIDDLE_FINGER_TIP
        "note": "E4",
        "file": "sounds/E4.mp3",
        "color": (255, 255, 0),  # Amarillo
        "hand": "left"
    },
    "indice_izq": {
        "landmark": 8,  # INDEX_FINGER_TIP
        "note": "F4",
        "file": "sounds/F4.mp3",
        "color": (0, 255, 0),  # Verde
        "hand": "left"
    },
    "pulgar_izq": {
        "landmark": 4,  # THUMB_TIP
        "note": "G4",
        "file": "sounds/G4.mp3",
        "color": (255, 165, 0),  # Naranja
        "hand": "left"
    },

    # MANO DERECHA (dedos más agudos)
    "pulgar_der": {
        "landmark": 4,  # THUMB_TIP (mano derecha)
        "note": "A4",
        "file": "sounds/A4.mp3",
        "color": (255, 0, 0),  # Rojo
        "hand": "right"
    },
    "indice_der": {
        "landmark": 8,  # INDEX_FINGER_TIP
        "note": "B4",
        "file": "sounds/B4.mp3",
        "color": (0, 0, 255),  # Azul
        "hand": "right"
    },
    "medio_der": {
        "landmark": 12,  # MIDDLE_FINGER_TIP
        "note": "C5",
        "file": "sounds/C5.mp3",
        "color": (255, 192, 203),  # Rosa
        "hand": "right"
    },
    "anular_der": {
        "landmark": 16,  # RING_FINGER_TIP
        "note": "D5",
        "file": "sounds/D5.mp3",
        "color": (0, 128, 0),  # Verde oscuro
        "hand": "right"
    },
    "menique_der": {
        "landmark": 20,  # PINKY_TIP
        "note": "E5",
        "file": "sounds/E5.mp3",
        "color": (255, 20, 147),  # Rosa oscuro
        "hand": "right"
    }
}


def generate_sine_wave(frequency, duration=0.8, sample_rate=44100, volume=0.3):
    """Genera un tono sinusoidal como fallback"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = np.sin(2 * np.pi * frequency * t)
    audio = np.int16(wave * 32767 * volume)
    return pygame.sndarray.make_sound(audio)


def load_note_sounds():
    """Carga los samples de audio reales"""
    sounds = {}
    # Frecuencias para las notas (Hz)
    frequencies = {
        "C4": 261.63, "D4": 293.66, "E4": 329.63, "F4": 349.23, "G4": 392.00,
        "A4": 440.00, "B4": 493.88, "C5": 523.25, "D5": 587.33, "E5": 659.25
    }

    for finger, config in NOTES_CONFIG.items():
        try:
            if os.path.exists(config["file"]):
                sounds[finger] = pygame.mixer.Sound(config["file"])
                print(f"✓ Cargado: {config['file']}")
            else:
                # Generar tono de fallback con frecuencia correcta
                freq = frequencies.get(config["note"], 440)
                sounds[finger] = generate_sine_wave(freq)
                print(f"✗ Usando tono generado para: {config['note']}")
        except Exception as e:
            print(f"Error cargando {config['file']}: {e}")
            freq = frequencies.get(config["note"], 440)
            sounds[finger] = generate_sine_wave(freq)
    return sounds


note_sounds = load_note_sounds()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # ¡Ahora detecta 2 manos!
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Estado de los dedos
finger_states = {finger: False for finger in NOTES_CONFIG.keys()}
activation_threshold = 0.15  # Umbral de activación


def get_finger_base_landmark(finger_name, landmark_idx):
    """Obtiene el landmark del nudillo base para cada dedo"""
    if "pulgar" in finger_name:
        return landmark_idx - 2  # THUMB_IP
    else:
        return landmark_idx - 2  # MCP joint


def draw_finger_info(frame, hand_landmarks, finger_name, config, handedness):
    """Dibuja información y estado de cada dedo"""
    landmark_idx = config["landmark"]
    color = config["color"]

    # Obtener coordenadas del dedo
    x = int(hand_landmarks.landmark[landmark_idx].x * screen_width)
    y = int(hand_landmarks.landmark[landmark_idx].y * screen_height)

    # Obtener coordenadas del nudillo base
    base_landmark = get_finger_base_landmark(finger_name, landmark_idx)
    base_y = int(hand_landmarks.landmark[base_landmark].y * screen_height)

    # Calcular posición relativa
    finger_length = max(1, y - base_y)
    current_position = y - base_y
    activation_pixels = finger_length * activation_threshold

    # Verificar si está activado
    is_active = current_position > activation_pixels

    # Dibujar punto del dedo (más grande)
    cv2.circle(frame, (x, y), 18, color, -1)
    cv2.circle(frame, (x, y), 18, (255, 255, 255), 3)

    # Dibujar línea de referencia del umbral
    threshold_y = base_y + int(activation_pixels)
    cv2.line(frame, (x - 30, threshold_y), (x + 30, threshold_y), (255, 255, 255), 3)

    # Dibujar información de estado
    status = "🎵 ACTIVO!" if is_active else f"{int((current_position / finger_length) * 100)}%"
    status_color = (0, 255, 0) if is_active else color

    # Nombre de la nota
    cv2.putText(frame, config["note"], (x - 25, y - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

    # Estado
    cv2.putText(frame, status, (x - 40, y + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    return is_active, current_position / finger_length


def process_hand_gestures(frame, hand_landmarks, handedness):
    """Procesa los gestos de la mano y activa las notas"""
    # Dibujar landmarks completos de la mano
    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Determinar si es mano izquierda o derecha
    is_right_hand = handedness.classification[0].label == "Right"
    hand_type = "right" if is_right_hand else "left"

    # Procesar cada dedo de esta mano
    active_fingers = []

    for finger_name, config in NOTES_CONFIG.items():
        if config["hand"] == hand_type:
            try:
                is_active, position = draw_finger_info(frame, hand_landmarks, finger_name, config, handedness)

                if is_active:
                    active_fingers.append(finger_name)

                    # Reproducir sonido si no estaba activo antes
                    if not finger_states[finger_name]:
                        note_sounds[finger_name].stop()
                        note_sounds[finger_name].play()
                        finger_states[finger_name] = True
                        print(f"🎹 {config['note']} ({hand_type}) activada!")
                else:
                    finger_states[finger_name] = False

            except Exception as e:
                print(f"Error procesando {finger_name}: {e}")
                continue

    return active_fingers


def draw_ui(frame, all_active_fingers):
    """Dibuja la interfaz de usuario"""
    # Panel de información
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (600, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "🎹 PIANO VIRTUAL - 10 DEDOS (2 MANOS)",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.putText(frame, "Instrucciones: Baja la punta de cada dedo para tocar notas",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Mostrar dedos activos
    if all_active_fingers:
        active_notes = [NOTES_CONFIG[f]["note"] for f in all_active_fingers]
        active_text = "🎶 Tocando: " + ", ".join(active_notes)
        cv2.putText(frame, active_text, (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Leyenda de MANO IZQUIERDA
    cv2.putText(frame, "Mano Izquierda:", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    y_offset = 150
    left_notes = ["C4", "D4", "E4", "F4", "G4"]
    left_colors = [(128, 0, 128), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 165, 0)]

    for i, note in enumerate(left_notes):
        cv2.circle(frame, (130, y_offset), 6, left_colors[i], -1)
        cv2.putText(frame, f"{note}", (145, y_offset + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20

    # Leyenda de MANO DERECHA
    cv2.putText(frame, "Mano Derecha:", (250, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    y_offset = 150
    right_notes = ["A4", "B4", "C5", "D5", "E5"]
    right_colors = [(255, 0, 0), (0, 0, 255), (255, 192, 203), (0, 128, 0), (255, 20, 147)]

    for i, note in enumerate(right_notes):
        cv2.circle(frame, (360, y_offset), 6, right_colors[i], -1)
        cv2.putText(frame, f"{note}", (375, y_offset + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20


# Bucle principal
print("Iniciando Piano Virtual - 10 DEDOS")
print("Notas disponibles: C4, D4, E4, F4, G4, A4, B4, C5, D5, E5")
print("Coloca AMBAS manos frente a la cámara!")

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (screen_width, screen_height))
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        all_active_fingers = []

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[hand_idx]
                try:
                    active_fingers = process_hand_gestures(frame, hand_landmarks, handedness)
                    all_active_fingers.extend(active_fingers)
                except Exception as e:
                    print(f"Error procesando mano: {e}")
                    continue

        draw_ui(frame, all_active_fingers)

        # Mostrar frame
        cv2.imshow("🎹 Piano Virtual - 10 DEDOS (Presiona 'Q' para salir)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    print("Cerrando aplicación...")
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()