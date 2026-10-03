"""
Staircase & Elevation Step Periodic Gradient Detector.
Detects Stairs Up and Stairs Down by analyzing horizontal riser lines and periodic vertical spacing.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

try:
    from . import config
except ImportError:
    import config

class StaircaseHazard:
    def __init__(self,
                 box_xyxy: Tuple[int, int, int, int],
                 stair_type: str,  # 'Stairs Up' or 'Stairs Down'
                 line_count: int,
                 confidence: float,
                 distance_meters: float,
                 lines: List[Tuple[int, int, int, int]]):
        self.box_xyxy = box_xyxy
        self.stair_type = stair_type
        self.line_count = line_count
        self.confidence = confidence
        self.distance_meters = distance_meters
        self.lines = lines

class StaircaseDetector:
    def __init__(self):
        self.camera_height_m = 1.45

    def detect_staircase(self, frame: np.ndarray) -> Optional[StaircaseHazard]:
        """
        Scans mid-ground walking corridor for periodic horizontal stair edges.
        """
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        y_min = int(h * config.MID_STAIR_ZONE_MIN_Y)
        y_max = int(h * config.MID_STAIR_ZONE_MAX_Y)
        stair_roi = frame[y_min:y_max, 0:w]

        if stair_roi.size == 0:
            return None

        gray = cv2.cvtColor(stair_roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 1. Horizontal Sobel filter to accentuate horizontal step risers
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobel_y = np.uint8(np.absolute(sobel_y))

        # 2. Canny edge detector on horizontal gradients
        edges = cv2.Canny(abs_sobel_y, 40, 120)

        # 3. Probabilistic Hough Line Transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=50,
            minLineLength=config.STAIR_MIN_LINE_LENGTH,
            maxLineGap=config.STAIR_MAX_LINE_GAP
        )

        if lines is None or len(lines) == 0:
            return None

        # Filter for strictly near-horizontal lines
        horizontal_lines = []
        for line in lines:
            coords = np.array(line).flatten()
            if len(coords) < 4:
                continue
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            dx = x2 - x1
            dy = y2 - y1
            angle = np.degrees(np.arctan2(abs(dy), max(1, abs(dx))))

            if angle <= config.STAIR_MAX_LINE_ANGLE_DEG:
                # Convert coordinates to full frame
                horizontal_lines.append((x1, y_min + y1, x2, y_min + y2, (y1 + y2) / 2.0))

        if len(horizontal_lines) < config.STAIR_MIN_PARALLEL_LINES:
            return None

        # Sort lines vertically by Y coordinate
        horizontal_lines.sort(key=lambda l: l[4])

        # 4. Vertical periodicity test (check if spacing between step lines is consistent)
        y_centers = [l[4] for l in horizontal_lines]
        spacings = [y_centers[i+1] - y_centers[i] for i in range(len(y_centers)-1)]
        valid_spacings = [s for s in spacings if 10 <= s <= 70]

        if len(valid_spacings) < (config.STAIR_MIN_PARALLEL_LINES - 1):
            return None

        # Calculate bounding box enclosing all detected stair lines
        all_x = [l[0] for l in horizontal_lines] + [l[2] for l in horizontal_lines]
        all_y = [l[1] for l in horizontal_lines] + [l[3] for l in horizontal_lines]
        box_x1 = max(0, min(all_x) - 10)
        box_x2 = min(w, max(all_x) + 10)
        box_y1 = max(0, min(all_y) - 10)
        box_y2 = min(h, max(all_y) + 10)

        # 5. Classify Stairs Up vs. Stairs Down:
        # Stairs Up has converging lines at the top; Stairs Down has lower drop-off shadow intensity
        top_y_crop = gray[max(0, int(box_y1 - y_min)):min(stair_roi.shape[0], int(box_y1 - y_min + 30)), :]
        bot_y_crop = gray[max(0, int(box_y2 - y_min - 30)):min(stair_roi.shape[0], int(box_y2 - y_min)), :]
        
        top_mean = float(np.mean(top_y_crop)) if top_y_crop.size > 0 else 128.0
        bot_mean = float(np.mean(bot_y_crop)) if bot_y_crop.size > 0 else 128.0

        stair_type = "Stairs Down" if bot_mean < (top_mean * 0.85) else "Stairs Up"

        # Confidence calculation
        line_score = min(1.0, len(horizontal_lines) / 6.0)
        spacing_consistency = 1.0 - (np.std(valid_spacings) / (np.mean(valid_spacings) + 1e-5))
        confidence = round(float((0.60 * line_score) + (0.40 * max(0.0, spacing_consistency))), 2)

        if confidence < config.STAIR_CONFIDENCE_THRESHOLD:
            return None

        # Distance estimation to first step (bottom line)
        lowest_line_y = max(all_y)
        y_horizon = int(h * 0.45)
        y_diff = max(15, lowest_line_y - y_horizon)
        est_dist = (self.camera_height_m * config.FOCAL_LENGTH_PIXELS) / float(y_diff)
        est_dist = round(max(0.6, min(10.0, est_dist)), 1)

        extracted_lines = [(l[0], l[1], l[2], l[3]) for l in horizontal_lines]

        return StaircaseHazard(
            box_xyxy=(box_x1, box_y1, box_x2, box_y2),
            stair_type=stair_type,
            line_count=len(horizontal_lines),
            confidence=confidence,
            distance_meters=est_dist,
            lines=extracted_lines
        )
