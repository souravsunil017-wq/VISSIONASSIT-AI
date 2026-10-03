"""
Configuration settings for Feature 4: Familiar Person & Face Recognition.
"""

import os

# Base paths
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(MODULE_DIR, ".."))
MODELS_DIR = os.path.join(MODULE_DIR, "models")

# Known faces dataset directory (checks root /known_faces first, then module /known_faces)
ROOT_KNOWN_FACES = os.path.join(PROJECT_ROOT, "known_faces")
LOCAL_KNOWN_FACES = os.path.join(MODULE_DIR, "known_faces")
KNOWN_FACES_DIR = ROOT_KNOWN_FACES if os.path.exists(ROOT_KNOWN_FACES) else LOCAL_KNOWN_FACES

# Cached vector database path
DATABASE_CACHE_PATH = os.path.join(MODULE_DIR, "face_database.pkl")

# Pretrained ONNX Models
YUNET_MODEL_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

# Face Detection Parameters
DETECTION_CONF_THRESHOLD = 0.65
NMS_THRESHOLD = 0.3
TOP_K = 10

# Face Recognition Parameters (Cosine Similarity)
# SFace Cosine Similarity: 0.363 is official threshold; 0.40+ gives high confidence precision.
COSINE_SIMILARITY_THRESHOLD = 0.42

# Spatial Zones (Horizontal ratios)
ZONE_LEFT_LIMIT = 0.33
ZONE_RIGHT_LIMIT = 0.67

# Distance Estimation Constants (Estimated focal length and real-world face height)
REAL_FACE_HEIGHT_METERS = 0.20  # ~20 cm average human face height
FOCAL_LENGTH_PIXELS = 550.0     # Calibrated for 640x480 resolution

# Speech timing / cooldowns
SPEECH_COOLDOWN_PER_PERSON_SEC = 8.0  # Avoid repeating the same recognized person within 8s
UNKNOWN_PERSON_COOLDOWN_SEC = 10.0    # Cooldown for unknown alert

# Display Resolution
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
