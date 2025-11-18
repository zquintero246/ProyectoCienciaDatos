import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time


BaseOptions = python.BaseOptions
RunningMode = vision.RunningMode

HAND_MODEL = "models/hand_landmarker.task"
POSE_MODEL = "models/pose_landmarker_full.task"
FACE_MODEL = "models/face_landmarker.task"

mp_image_format = mp.ImageFormat.SRGB



def hand_callback(result, output_image, timestamp_ms):
    print("\nMANOS detectadas:", len(result.hand_landmarks))
    # result.hand_landmarks → lista de 21 puntos por mano

def pose_callback(result, output_image, timestamp_ms):
    if result.pose_landmarks:
        print("\nPOSE detectada (33 puntos)")

def face_callback(result, output_image, timestamp_ms):
    if result.face_landmarks:
        print("\nROSTRO detectado")




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

face_options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_MODEL),
    running_mode=RunningMode.LIVE_STREAM,
    result_callback=face_callback,
    output_face_blendshapes=False
)




cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

with vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker, \
     vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
     vision.FaceLandmarker.create_from_options(face_options) as face_landmarker:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("LSC detector", frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp_image_format, data=rgb_frame)

        timestamp_ms = int(time.time() * 1000)

        hand_landmarker.detect_async(mp_image, timestamp_ms)
        pose_landmarker.detect_async(mp_image, timestamp_ms)
        face_landmarker.detect_async(mp_image, timestamp_ms)

        

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
