"""
Face Detection and Deep Metric Embedding Extractor.
Uses OpenCV YuNet (Face Detector) and SFace (128-D Deep Feature Embedding).
Supports JPG, PNG, JPEG, and iPhone HEIC photo formats.
"""

import os
import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_ENABLED = True
except Exception:
    HEIF_ENABLED = False

try:
    from . import config
except ImportError:
    import config

@dataclass
class DetectedFace:
    box: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    landmarks: np.ndarray          # 5 facial landmarks (eyes, nose, mouth corners)
    raw_face_data: np.ndarray      # 15-element array from YuNet
    feature_vector: Optional[np.ndarray] = None
    identity: str = "Unknown"
    match_score: float = 0.0
    distance_meters: float = 0.0
    zone: str = "CENTER"           # 'LEFT', 'CENTER', 'RIGHT'

def load_image_robust(file_path: str) -> Optional[np.ndarray]:
    """
    Robust image loader supporting standard formats (JPG, PNG) and iPhone HEIC files.
    Returns BGR numpy array compatible with OpenCV.
    """
    if not os.path.exists(file_path):
        return None

    ext = os.path.splitext(file_path)[1].lower()
    
    # 1. Standard OpenCV loader for typical formats
    if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        img = cv2.imread(file_path)
        if img is not None:
            return img

    # 2. Pillow + pillow_heif fallback for HEIC and exotic formats
    try:
        pil_img = Image.open(file_path)
        pil_img = pil_img.convert("RGB")
        rgb_arr = np.array(pil_img)
        bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        return bgr_arr
    except Exception as e:
        print(f"[Image Loader] Warning: Could not load {file_path}: {e}")
        return None

class FaceEngine:
    def __init__(self, input_size: Tuple[int, int] = (config.FRAME_WIDTH, config.FRAME_HEIGHT)):
        self.input_size = input_size
        self._init_models()

    def _init_models(self):
        """Initializes OpenCV YuNet Face Detector and SFace Recognizer."""
        if not os.path.exists(config.YUNET_MODEL_PATH) or not os.path.exists(config.SFACE_MODEL_PATH):
            raise FileNotFoundError(f"Face model files not found in {config.MODELS_DIR}.")

        # 1. Face Detector (YuNet)
        self.detector = cv2.FaceDetectorYN.create(
            model=config.YUNET_MODEL_PATH,
            config="",
            input_size=self.input_size,
            score_threshold=config.DETECTION_CONF_THRESHOLD,
            nms_threshold=config.NMS_THRESHOLD,
            top_k=config.TOP_K
        )

        # 2. Face Recognizer (SFace)
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=config.SFACE_MODEL_PATH,
            config=""
        )

    def set_input_size(self, size: Tuple[int, int]):
        """Dynamically updates detector input resolution if frame size changes."""
        if self.input_size != size:
            self.input_size = size
            self.detector.setInputSize(size)

    def detect_and_extract_faces(self, frame: np.ndarray) -> List[DetectedFace]:
        """
        Detects all faces in the frame and extracts aligned 128-D feature embeddings.
        """
        h, w = frame.shape[:2]
        self.set_input_size((w, h))

        _, raw_faces = self.detector.detect(frame)
        if raw_faces is None:
            return []

        detected_faces = []
        for raw_face in raw_faces:
            # YuNet output layout:
            # [0:4] = x, y, w, h
            # [4:14] = 5 landmarks (x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm)
            # [14] = confidence score
            box = tuple(map(int, raw_face[0:4]))
            conf = float(raw_face[14])
            landmarks = raw_face[4:14].reshape((5, 2))

            # Crop & Align Face to canonical orientation (112x112)
            aligned_face = self.recognizer.alignCrop(frame, raw_face)
            
            # Extract normalized 128-D feature vector
            feature = self.recognizer.feature(aligned_face)

            # Compute spatial zone (Left, Center, Right)
            cx = box[0] + (box[2] // 2)
            ratio_x = cx / max(1, w)
            if ratio_x < config.ZONE_LEFT_LIMIT:
                zone = "LEFT"
            elif ratio_x > config.ZONE_RIGHT_LIMIT:
                zone = "RIGHT"
            else:
                zone = "CENTER"

            # Compute estimated distance
            face_h = max(10, box[3])
            distance = round((config.REAL_FACE_HEIGHT_METERS * config.FOCAL_LENGTH_PIXELS) / face_h, 1)
            distance = max(0.4, min(distance, 6.0))

            face_obj = DetectedFace(
                box=box,
                confidence=conf,
                landmarks=landmarks,
                raw_face_data=raw_face,
                feature_vector=feature,
                distance_meters=distance,
                zone=zone
            )
            detected_faces.append(face_obj)

        return detected_faces

    def extract_single_face_vector(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects the primary face in an enrollment photo and returns its 128-D embedding vector.
        """
        h, w = image.shape[:2]
        self.set_input_size((w, h))

        _, raw_faces = self.detector.detect(image)
        if raw_faces is None or len(raw_faces) == 0:
            return None

        # Pick the largest face in enrollment image
        best_face = max(raw_faces, key=lambda f: f[2] * f[3])
        aligned_face = self.recognizer.alignCrop(image, best_face)
        feature = self.recognizer.feature(aligned_face)
        return feature

    def compute_cosine_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Computes Cosine Similarity between two 128-D face vectors.
        Score ranges from -1.0 to 1.0 (typically 0.363+ is considered a match in SFace).
        """
        score = self.recognizer.match(feat1, feat2, cv2.FaceRecognizerSF_FR_COSINE)
        return float(score)
