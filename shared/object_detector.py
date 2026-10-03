"""
Universal High-Speed Object Detector using OpenCV DNN.
Runs natively without PyTorch DLL or Windows Security issues.
Detects 90+ COCO classes (chair, dining table, bottle, person, car, bus, etc.) with real-time performance.
"""

import os
import sys
import cv2
import numpy as np
from typing import List, Dict, Any

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
PB_PATH = os.path.join(MODEL_DIR, "frozen_inference_graph.pb")
PBTXT_PATH = os.path.join(MODEL_DIR, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
COCO_NAMES_PATH = os.path.join(MODEL_DIR, "coco.names")

class UniversalObjectDetector:
    """
    Robust Object Detector using OpenCV DNN DetectionModel.
    Zero external dependencies, immune to PyTorch Windows Security / DLL blocks.
    """
    def __init__(self, conf_threshold: float = 0.40, nms_threshold: float = 0.40):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.net = None
        self.class_names = []
        self._load_model()

    def _load_model(self):
        try:
            if not os.path.exists(PB_PATH) or not os.path.exists(PBTXT_PATH) or not os.path.exists(COCO_NAMES_PATH):
                print(f"[Detector Error] Model files missing in {MODEL_DIR}")
                return

            with open(COCO_NAMES_PATH, "rt", encoding="utf-8") as f:
                self.class_names = [line.strip() for line in f if line.strip()]

            self.net = cv2.dnn_DetectionModel(PB_PATH, PBTXT_PATH)
            self.net.setInputSize(320, 320)
            self.net.setInputScale(1.0 / 127.5)
            self.net.setInputMean((127.5, 127.5, 127.5))
            self.net.setInputSwapRB(True)
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            except Exception:
                pass
            print(f"[UniversalDetector] Initialized SSD MobileNet V3 ({len(self.class_names)} classes).")
        except Exception as e:
            print(f"[UniversalDetector Error] Failed to initialize: {e}")
            self.net = None

    def is_ready(self) -> bool:
        return self.net is not None

    def detect(self, frame: np.ndarray, conf_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Runs object detection on a BGR image frame.
        Returns list of dicts: {'box': (x1, y1, x2, y2), 'class_name': str, 'conf': float}
        """
        if self.net is None or frame is None or frame.size == 0:
            return []

        thresh = conf_threshold if conf_threshold is not None else self.conf_threshold
        
        try:
            classes, confidences, boxes = self.net.detect(
                frame,
                confThreshold=thresh,
                nmsThreshold=self.nms_threshold
            )
        except Exception as e:
            print(f"[Detector Runtime Error]: {e}")
            return []

        detections = []
        if len(classes) == 0:
            return detections

        for classId, conf, box in zip(classes.flatten(), confidences.flatten(), boxes):
            x, y, w, h = box
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(frame.shape[1], int(x + w))
            y2 = min(frame.shape[0], int(y + h))

            # 1-indexed mapping for TensorFlow COCO model
            idx = int(classId) - 1
            if 0 <= idx < len(self.class_names):
                cls_name = self.class_names[idx]
            else:
                cls_name = f"object_{classId}"

            # Remap technical terms to clear everyday names for speech
            if cls_name == "dining table":
                cls_name = "table"

            detections.append({
                'box': (x1, y1, x2, y2),
                'class_name': cls_name,
                'conf': float(conf)
            })

        return detections
