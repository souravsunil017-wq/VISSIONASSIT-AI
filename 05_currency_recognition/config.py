"""
Configuration for Pre-Trained Deep Learning Indian Currency Recognition System.
Covers Mahatma Gandhi New Series: ₹10, ₹20, ₹50, ₹100, ₹200, ₹500.
"""

import os

# Base paths
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(MODULE_DIR, ".."))

# Currency notes dataset directories (scans both root currency_notes and module currency_dataset)
ROOT_CURRENCY_NOTES = os.path.join(PROJECT_ROOT, "currency_notes")
MODULE_CURRENCY_DATASET = os.path.join(MODULE_DIR, "currency_dataset")
LOCAL_CURRENCY_NOTES = os.path.join(MODULE_DIR, "currency_notes")
CURRENCY_DATASET_DIRS = [ROOT_CURRENCY_NOTES, MODULE_CURRENCY_DATASET, LOCAL_CURRENCY_NOTES]
CURRENCY_NOTES_DIR = MODULE_CURRENCY_DATASET if os.path.exists(MODULE_CURRENCY_DATASET) else ROOT_CURRENCY_NOTES

# Cached vector database path
DATABASE_CACHE_PATH = os.path.join(MODULE_DIR, "currency_database.pkl")

# Supported Denominations & RBI Specifications
DENOMINATIONS = ["10", "20", "50", "100", "200", "500"]

DENOMINATION_INFO = {
    "10": {
        "name": "10 Rupees",
        "symbol": "₹10",
        "base_color": "Chocolate Brown",
        "motif": "Konark Sun Temple",
        "nominal_size_mm": (123, 63),
        "aspect_ratio": 1.95
    },
    "20": {
        "name": "20 Rupees",
        "symbol": "₹20",
        "base_color": "Greenish Yellow",
        "motif": "Ellora Caves",
        "nominal_size_mm": (129, 63),
        "aspect_ratio": 2.05
    },
    "50": {
        "name": "50 Rupees",
        "symbol": "₹50",
        "base_color": "Fluorescent Blue",
        "motif": "Hampi with Chariot",
        "nominal_size_mm": (135, 66),
        "aspect_ratio": 2.05
    },
    "100": {
        "name": "100 Rupees",
        "symbol": "₹100",
        "base_color": "Lavender / Purple",
        "motif": "Rani ki Vav",
        "nominal_size_mm": (142, 66),
        "aspect_ratio": 2.15
    },
    "200": {
        "name": "200 Rupees",
        "symbol": "₹200",
        "base_color": "Bright Orange-Yellow",
        "motif": "Sanchi Stupa",
        "nominal_size_mm": (146, 66),
        "aspect_ratio": 2.21
    },
    "500": {
        "name": "500 Rupees",
        "symbol": "₹500",
        "base_color": "Stone Grey",
        "motif": "Red Fort",
        "nominal_size_mm": (150, 66),
        "aspect_ratio": 2.27
    }
}

# Recognition Thresholds & Anti-Flicker Filtering
CONFIDENCE_ACCEPT_THRESHOLD = 0.80   # Requires 80% neural confidence to confirm note
TEMPORAL_CONSISTENCY_FRAMES = 4     # Number of consecutive frames needed to prevent false positives
ANNOUNCEMENT_COOLDOWN = 3.5          # Seconds before re-announcing same note
VOICE_RATE = 1                       # SAPI voice rate
VOICE_VOLUME = 100                   # SAPI volume
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
