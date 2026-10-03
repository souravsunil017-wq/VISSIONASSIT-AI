"""
Overhead Eye-Level Hazard & Wet Surface Glint Detector.
Spots low-hanging tree branches, low beams, scaffolding at head height, and wet slippery surfaces.
"""

import cv2
import numpy as np
from typing import Tuple, Optional

try:
    from . import config
except ImportError:
    import config

class OverheadHazard:
    def __init__(self,
                 box_xyxy: Tuple[int, int, int, int],
                 hazard_name: str,
                 edge_density: float,
                 confidence: float):
        self.box_xyxy = box_xyxy
        self.hazard_name = hazard_name
        self.edge_density = edge_density
        self.confidence = confidence

class WetSurfaceHazard:
    def __init__(self,
                 box_xyxy: Tuple[int, int, int, int],
                 glint_pixel_count: int,
                 confidence: float):
        self.box_xyxy = box_xyxy
        self.glint_pixel_count = glint_pixel_count
        self.confidence = confidence

class OverheadAndGlintDetector:
    def __init__(self):
        pass

    def detect_overhead_hazard(self, frame: np.ndarray) -> Optional[OverheadHazard]:
        """
        Scans upper eye-level ROI (top 40% of frame) for low-hanging branches, beams, or scaffolding.
        """
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        y_end = int(h * config.OVERHEAD_ZONE_MAX_Y)
        
        # Center walking corridor at head height
        x_start = int(w * 0.20)
        x_end = int(w * 0.80)
        overhead_roi = frame[0:y_end, x_start:x_end]

        if overhead_roi.size == 0:
            return None

        gray = cv2.cvtColor(overhead_roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detector for branch / beam texture
        edges = cv2.Canny(blurred, 50, 150)
        total_pixels = edges.size
        edge_pixels = int(np.count_nonzero(edges))
        edge_density = edge_pixels / float(max(1, total_pixels))

        if edge_density >= config.OVERHEAD_EDGE_DENSITY_THRESHOLD:
            # High edge density in upper central corridor indicates low branch / overhead obstacle
            confidence = min(1.0, edge_density / (config.OVERHEAD_EDGE_DENSITY_THRESHOLD * 2.0))
            box_xyxy = (x_start, 10, x_end, y_end)
            return OverheadHazard(
                box_xyxy=box_xyxy,
                hazard_name="Low-Hanging Obstacle / Branch",
                edge_density=round(edge_density, 3),
                confidence=round(confidence, 2)
            )

        return None

    def detect_wet_surface(self, frame: np.ndarray) -> Optional[WetSurfaceHazard]:
        """
        Scans floor plane for high-luminance specular reflections / liquid glint.
        """
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        y_start = int(h * config.GROUND_FLOOR_ZONE_MIN_Y)
        floor_roi = frame[y_start:h, 0:w]

        if floor_roi.size == 0:
            return None

        gray = cv2.cvtColor(floor_roi, cv2.COLOR_BGR2GRAY)
        
        # Threshold for extreme specular highlight glint
        _, glint_mask = cv2.threshold(gray, config.WET_SURFACE_GLINT_THRESHOLD, 255, cv2.THRESH_BINARY)
        glint_count = int(np.count_nonzero(glint_mask))

        if glint_count >= config.WET_SURFACE_MIN_PIXELS:
            confidence = min(1.0, glint_count / (config.WET_SURFACE_MIN_PIXELS * 3.0))
            contours, _ = cv2.findContours(glint_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(largest)
                return WetSurfaceHazard(
                    box_xyxy=(x, y_start + y, x + bw, y_start + y + bh),
                    glint_pixel_count=glint_count,
                    confidence=round(confidence, 2)
                )

        return None
