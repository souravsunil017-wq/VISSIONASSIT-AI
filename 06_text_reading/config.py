"""
Configuration for Feature 6: Text Reading & Audio Narration System.
"""

# OCR & Detection Settings
OCR_LANGUAGES = ['en']            # Languages to detect (English)
CONFIDENCE_THRESHOLD = 0.40       # Minimum confidence to accept recognized text
MIN_TEXT_AREA_PX = 400            # Minimum bounding box area for valid text line
USE_GPU = False                   # Run on CPU for universal compatibility

# Live Spotter Settings (Mode 1)
LIVE_SPOTTER_INTERVAL_SEC = 1.8   # Interval between background live text spots
MAX_LIVE_ANNOUNCE_WORDS = 8       # Max words to announce in live signboard mode

# Reading Voice Settings
VOICE_RATE = 1                    # SAPI speech rate (1 is natural clear reading speed)
VOICE_VOLUME = 100                # SAPI speech volume (0 - 100)
SENTENCE_PAUSE_SEC = 0.4          # Pause duration between sentences in document reading mode

# Image Preprocessing
PREPROCESS_CONTRAST_CLIP = 2.0    # CLAHE contrast limit
DENOISE_KERNEL_SIZE = 3           # Gaussian blur kernel size for background noise removal
