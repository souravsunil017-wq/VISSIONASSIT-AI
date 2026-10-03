"""
Configuration for Feature 3: Obstacle & Path-Hazard Detection System.
Thresholds and ROI zone partitions for Ground Drops, Potholes, Stairs, Overhead Hazards, and Slick Surfaces.
"""

import os

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(MODULE_DIR, ".."))

# Camera Calibration
FOCAL_LENGTH_PIXELS = 550.0  # Calibrated focal length for standard 640x480 webcam / smartphone FOV
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Multi-Zone ROI Vertical Partitions (Fraction of Frame Height 0.0 - 1.0)
OVERHEAD_ZONE_MAX_Y = 0.40   # 0.00 to 0.40 = Overhead / Head-Height Zone
MID_STAIR_ZONE_MIN_Y = 0.30  # 0.30 to 0.75 = Mid-Ground Staircase Corridor
MID_STAIR_ZONE_MAX_Y = 0.80
GROUND_FLOOR_ZONE_MIN_Y = 0.55  # 0.55 to 1.00 = Ground / Floor Hazard Plane

# Spatial Lateral Corridors (Fraction of Frame Width 0.0 - 1.0)
CORRIDOR_LEFT_MAX = 0.33
CORRIDOR_CENTER_MAX = 0.67

# Pothole & Ground Drop-off Parameters
POTHOLE_MIN_CONTOUR_AREA = 1200      # Minimum pixel area to consider as pothole / cavity
POTHOLE_MAX_CONTOUR_AREA = 45000     # Maximum area
POTHOLE_DARKNESS_RATIO = 0.78        # Cavity mean luminance must be < 78% of surrounding floor
POTHOLE_CONFIDENCE_THRESHOLD = 0.55  # Minimum score to trigger alert

# Staircase Periodic Line Parameters
STAIR_MIN_PARALLEL_LINES = 3         # Minimum horizontal step lines needed to confirm staircase
STAIR_MAX_LINE_ANGLE_DEG = 18.0      # Maximum deviation from horizontal (0 degrees)
STAIR_MIN_LINE_LENGTH = 70           # Minimum pixel length of step riser line
STAIR_MAX_LINE_GAP = 25              # Maximum gap between step line segments
STAIR_CONFIDENCE_THRESHOLD = 0.60

# Overhead Hazard Parameters
OVERHEAD_EDGE_DENSITY_THRESHOLD = 0.12  # Fraction of edge pixels in overhead zone to trigger alert
OVERHEAD_CONFIDENCE_THRESHOLD = 0.50

# Wet / Slick Surface Parameters
WET_SURFACE_GLINT_THRESHOLD = 235    # High luminance reflection pixel threshold
WET_SURFACE_MIN_PIXELS = 1500

# Audio Announcement Cooldowns (seconds)
ALERT_COOLDOWN_POTHOLE = 2.5
ALERT_COOLDOWN_STAIRS = 3.0
ALERT_COOLDOWN_OVERHEAD = 2.5
ALERT_COOLDOWN_WET = 4.0
VOICE_RATE = 1
VOICE_VOLUME = 100
