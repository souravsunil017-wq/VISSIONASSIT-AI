"""
Configuration for Feature 2: Approaching Vehicle Detection & Time-to-Collision (TTC) System.
Calibrations for road vehicles, optical expansion thresholds, spatial corridors, and audio cues.
"""

import os

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(MODULE_DIR, ".."))

# YOLO Model Path
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "yolov8n.pt")

# Target Vehicle Classes (COCO Class IDs & Names)
# COCO IDs: 2: car, 3: motorcycle, 5: bus, 7: truck, 1: bicycle
VEHICLE_CLASS_MAP = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Real-world Nominal Vehicle Physical Dimensions (meters) for Monocular Distance Estimation
NOMINAL_VEHICLE_HEIGHTS = {
    "car": 1.50,
    "bus": 3.20,
    "truck": 3.00,
    "motorcycle": 1.15,
    "bicycle": 1.05
}

# Monocular Camera Geometry Calibration
FOCAL_LENGTH_PIXELS = 550.0  # Calibrated for standard 640x480 webcam / smartphone FOV
CONFIDENCE_THRESHOLD = 0.40  # YOLO detection confidence threshold
IOU_TRACK_THRESHOLD = 0.35   # IOU threshold for vehicle tracking association

# Optical Expansion & Motion Analysis Parameters
HISTORY_BUFFER_SIZE = 8      # Number of recent frames stored per vehicle track
MIN_FRAMES_FOR_TTC = 3       # Minimum frames needed before computing velocity
EXPANSION_RATE_THRESHOLD = 0.06  # Bounding box area growth rate (fraction/sec) considered 'Approaching'
RECEDING_RATE_THRESHOLD = -0.05  # Bounding box shrinking rate considered 'Moving Away'

# Threat Level & Time-to-Collision (TTC) Boundaries (seconds & meters)
CRITICAL_TTC_SECONDS = 2.5       # TTC < 2.5s -> Red Critical Collision Alert
WARNING_TTC_SECONDS = 5.0        # TTC 2.5s - 5.0s -> Yellow Warning Alert
CRITICAL_DISTANCE_METERS = 3.5   # Approaching vehicle closer than 3.5m -> Critical
WARNING_DISTANCE_METERS = 8.0    # Approaching vehicle closer than 8.0m -> Warning

# Spatial Corridor Boundaries (Fraction of Frame Width 0.0 - 1.0)
CORRIDOR_LEFT_MAX = 0.35         # 0.00 to 0.35 = Left Corridor
CORRIDOR_CENTER_MAX = 0.65       # 0.35 to 0.65 = Center Path (Direct Impact Zone)
                                 # 0.65 to 1.00 = Right Corridor

# Speech Announcement Cooldowns (seconds)
ALERT_COOLDOWN_CRITICAL = 1.8    # Repeat critical alerts rapidly
ALERT_COOLDOWN_WARNING = 3.5     # Standard warning interval
VOICE_RATE = 1
VOICE_VOLUME = 100

# Video Frame Dimensions
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
