# Camera / Video Feed Settings
CAMERA_INDEX = 0             # Default webcam index (use 0 or path to video file)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30
FOCAL_LENGTH_PIXELS = 550.0  # Calibrated focal length for standard phone / webcam
CAMERA_HEIGHT_M = 1.30       # Chest-level / walking holding height

# Detection & AI Model Settings
CONFIDENCE_THRESHOLD = 0.38  # Confident detections only

# Navigation Safety Zones (Proportions of frame width 0.0 to 1.0)
ZONE_LEFT_LIMIT = 0.33       # Left 33% of frame (0.00 -> 0.33)
ZONE_RIGHT_LIMIT = 0.67      # Center corridor (0.33 -> 0.67), Right 33% (0.67 -> 1.00)
GROUND_HORIZON_RATIO = 0.35  # Approximate horizon line (35% from top)

# Distance Proximity Gating (Meters)
DISTANCE_STOP_THRESHOLD = 1.25    # Distance <= 1.25m in front triggers STOP
DISTANCE_STEER_THRESHOLD = 2.60   # Distance <= 2.60m in front triggers STEER
DISTANCE_IGNORE_THRESHOLD = 3.20  # Distance > 3.20m is considered far / clear

# Relevant Walking Obstacle Categories
MAJOR_OBSTACLE_CLASSES = {
    "chair", "table", "dining table", "person", "couch", "sofa", "bed",
    "bench", "desk", "tv", "door", "refrigerator", "potted plant", "suitcase",
    "backpack", "car", "bus", "truck", "motorcycle", "bicycle", "dog"
}

# Voice Feedback Settings
VOICE_RATE = 1               # Speech speed (1 is clear standard speed)
VOICE_VOLUME = 100           # Volume (0 to 100)
ANNOUNCEMENT_COOLDOWN = 2.4  # Minimum seconds between new spoken instructions
CRITICAL_ALERT_COOLDOWN = 1.2 # Cooldown for urgent 'STOP' warnings
