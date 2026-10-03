import os
import sys
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

# Add parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.object_detector import UniversalObjectDetector

try:
    from . import config
except ImportError:
    import config

class DetectedVehicle:
    """Container holding raw detection data for a single vehicle."""
    def __init__(self,
                 box_xyxy: Tuple[int, int, int, int],
                 class_id: int,
                 class_name: str,
                 confidence: float,
                 distance_meters: float,
                 corridor: str):
        self.box_xyxy = box_xyxy
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.distance_meters = distance_meters
        self.corridor = corridor
        
        # Computed spatial geometry
        x1, y1, x2, y2 = box_xyxy
        self.width = max(1, x2 - x1)
        self.height = max(1, y2 - y1)
        self.area = self.width * self.height
        self.center_x = int((x1 + x2) / 2)
        self.center_y = int((y1 + y2) / 2)

class VehicleDetector:
    def __init__(self, model_path: str = None):
        self.detector = UniversalObjectDetector(conf_threshold=config.CONFIDENCE_THRESHOLD)
        self.vehicle_classes = {"car", "bus", "truck", "motorcycle", "bicycle"}

    def detect_vehicles(self, frame: np.ndarray) -> List[DetectedVehicle]:
        """
        Runs object detection and filters for approaching road vehicles.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        raw_detections = self.detector.detect(frame, conf_threshold=config.CONFIDENCE_THRESHOLD)
        detected_vehicles = []

        for det in raw_detections:
            class_name = det['class_name']
            if class_name not in self.vehicle_classes and "car" not in class_name and "truck" not in class_name and "bus" not in class_name and "motorcycle" not in class_name and "bicycle" not in class_name:
                continue

            conf = det['conf']
            x1, y1, x2, y2 = det['box']

            # Clip coordinates to frame bounds
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            box_height = max(1, y2 - y1)
            nominal_height = config.NOMINAL_VEHICLE_HEIGHTS.get(class_name, 1.50)

            # Monocular Distance Estimation via inverse height law: Z = (f * H_real) / h_box
            distance_meters = (config.FOCAL_LENGTH_PIXELS * nominal_height) / box_height
            distance_meters = max(0.5, min(50.0, distance_meters))

            # Spatial Corridor Classification
            cx = (x1 + x2) / 2.0
            norm_cx = cx / float(w)

            if norm_cx < config.CORRIDOR_LEFT_MAX:
                corridor = "Left"
            elif norm_cx <= config.CORRIDOR_CENTER_MAX:
                corridor = "Center"
            else:
                corridor = "Right"

            detected_vehicles.append(DetectedVehicle(
                box_xyxy=(x1, y1, x2, y2),
                class_id=0,
                class_name=class_name,
                confidence=conf,
                distance_meters=round(distance_meters, 2),
                corridor=corridor
            ))

        return detected_vehicles
