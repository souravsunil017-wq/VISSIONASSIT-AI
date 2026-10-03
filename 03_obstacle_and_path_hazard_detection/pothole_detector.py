"""
Ground Drop-off, Pothole, Manhole & Road Cavity Detector.
Analyzes lower floor ROI using contrast disruption, cavity depth luminance, and contour compactness.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

try:
    from . import config
except ImportError:
    import config

class PotholeHazard:
    def __init__(self,
                 box_xyxy: Tuple[int, int, int, int],
                 confidence: float,
                 distance_meters: float,
                 corridor: str,
                 hazard_type: str = "Pothole"):
        self.box_xyxy = box_xyxy
        self.confidence = confidence
        self.distance_meters = distance_meters
        self.corridor = corridor
        self.hazard_type = hazard_type

class PotholeDetector:
    def __init__(self):
        self.camera_height_m = 1.45  # Typical chest-level / phone-holding height

    def detect_potholes(self, frame: np.ndarray) -> List[PotholeHazard]:
        """
        Scans lower floor zone for potholes, road trenches, and ground drop-offs.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        y_start = int(h * config.GROUND_FLOOR_ZONE_MIN_Y)
        floor_roi = frame[y_start:h, 0:w]

        if floor_roi.size == 0:
            return []

        gray = cv2.cvtColor(floor_roi, cv2.COLOR_BGR2GRAY)
        
        # 1. Bilateral filter to smooth texture while keeping sharp cavity edges
        blurred = cv2.bilateralFilter(gray, 7, 50, 50)
        mean_floor_brightness = max(1.0, float(np.mean(blurred)))

        # 2. Adaptive contrast segmentation for dark cavities
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 6
        )

        # Morphological opening to clean small noise speckles
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hazards = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < config.POTHOLE_MIN_CONTOUR_AREA or area > config.POTHOLE_MAX_CONTOUR_AREA:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = bw / float(max(1, bh))

            # Discard extreme vertical lines or banners
            if aspect_ratio < 0.35 or aspect_ratio > 3.8:
                continue

            # Check cavity mean brightness inside bounding box
            cavity_crop = blurred[y:y+bh, x:x+bw]
            cavity_mean = float(np.mean(cavity_crop))
            brightness_ratio = cavity_mean / mean_floor_brightness

            # True potholes are darker than surrounding flat floor plane
            if brightness_ratio > config.POTHOLE_DARKNESS_RATIO:
                continue

            # Circular / elliptical compactness metric
            perimeter = cv2.arcLength(cnt, True)
            compactness = (4.0 * np.pi * area) / max(1.0, perimeter * perimeter)

            # Confidence score calculation
            darkness_score = 1.0 - (brightness_ratio / config.POTHOLE_DARKNESS_RATIO)
            confidence = (0.65 * darkness_score) + (0.35 * min(1.0, compactness * 1.5))

            if confidence < config.POTHOLE_CONFIDENCE_THRESHOLD:
                continue

            # Global frame coordinates
            abs_y1 = y_start + y
            abs_y2 = y_start + y + bh
            abs_x1 = x
            abs_x2 = x + bw

            # Distance estimation based on bottom Y position on floor
            y_horizon = int(h * 0.45)
            y_diff = max(10, abs_y2 - y_horizon)
            est_dist = (self.camera_height_m * config.FOCAL_LENGTH_PIXELS) / float(y_diff)
            est_dist = round(max(0.5, min(12.0, est_dist)), 1)

            # Lateral corridor classification
            cx = (abs_x1 + abs_x2) / 2.0
            norm_cx = cx / float(w)
            if norm_cx < config.CORRIDOR_LEFT_MAX:
                corridor = "Left"
            elif norm_cx <= config.CORRIDOR_CENTER_MAX:
                corridor = "Center"
            else:
                corridor = "Right"

            hazards.append(PotholeHazard(
                box_xyxy=(abs_x1, abs_y1, abs_x2, abs_y2),
                confidence=round(confidence, 2),
                distance_meters=est_dist,
                corridor=corridor,
                hazard_type="Pothole / Drop-off"
            ))

        # Sort hazards by closest distance
        hazards.sort(key=lambda item: item.distance_meters)
        return hazards
