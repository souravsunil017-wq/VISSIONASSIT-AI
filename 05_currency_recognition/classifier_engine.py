"""
Deep Learning Classifier Engine for Indian Currency (INR) Recognition.
Loads pre-trained PyTorch MobileNetV3 model trained on Kaggle Indian Currency Dataset.
Includes Temporal Consistency Filter to completely eliminate false flicker & background triggers.
"""

import os
from collections import deque
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import Dict, Any, Optional

try:
    from . import config
except ImportError:
    import config

MODEL_PATH = os.path.join(os.path.dirname(__file__), "indian_currency_mobilenet.pth")

class CurrencyResult:
    def __init__(self, is_valid: bool = False, denomination: str = "Unknown",
                 confidence: float = 0.0, class_probs: Optional[Dict[str, float]] = None,
                 details: Optional[Dict[str, Any]] = None):
        self.is_valid_note = is_valid
        self.denomination = denomination
        self.final_confidence = confidence
        self.class_probs = class_probs or {}
        self.layer_details = details or {}

class MobileNetCurrencyClassifier:
    def __init__(self, model_path: str = MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ["10", "20", "50", "100", "200", "500", "background"]
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Temporal consistency FIFO queue to prevent single-frame flickering
        self.history_len = getattr(config, "TEMPORAL_CONSISTENCY_FRAMES", 4)
        self.prediction_history = deque(maxlen=self.history_len)

    def _load_model(self, model_path: str) -> nn.Module:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(256, len(self.classes))
        )
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            self.classes = checkpoint.get("classes", self.classes)
            print(f"[Model Loader] Loaded MobileNetV3 trained on Indian Currency Dataset: {model_path}")
        else:
            print(f"[Model Loader Warning] Weights file not found at {model_path}. Using base model.")

        model.to(self.device)
        model.eval()
        return model

    def predict(self, frame: np.ndarray) -> CurrencyResult:
        if frame is None or frame.size == 0:
            self.prediction_history.clear()
            return CurrencyResult(is_valid=False)

        h, w = frame.shape[:2]
        # Crop center focus area where user holds banknote
        cw, ch = int(w * 0.75), int(h * 0.70)
        cx, cy = int((w - cw) / 2), int((h - ch) / 2)
        roi = frame[cy:cy+ch, cx:cx+cw]

        # Convert BGR to RGB
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb_roi).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        class_probs = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
        
        # Sort predictions
        top_idx = int(np.argmax(probs))
        top_class = self.classes[top_idx]
        top_conf = float(probs[top_idx])

        # Add single frame prediction to temporal history
        if top_class != "background" and top_conf >= config.CONFIDENCE_ACCEPT_THRESHOLD:
            self.prediction_history.append((top_class, top_conf))
        else:
            self.prediction_history.append(("background", top_conf))

        # Temporal Consistency Filter: Must match for all frames in history buffer
        is_valid = False
        confirmed_denom = "Unknown"
        confirmed_conf = 0.0

        if len(self.prediction_history) == self.history_len:
            # Check if all recent frames agree on the same banknote
            classes_in_buffer = [item[0] for item in self.prediction_history]
            if classes_in_buffer.count(top_class) >= (self.history_len - 1) and top_class != "background":
                is_valid = True
                confirmed_denom = top_class
                confirmed_conf = sum(item[1] for item in self.prediction_history if item[0] == top_class) / max(1, classes_in_buffer.count(top_class))

        details = {}
        if is_valid and confirmed_denom in config.DENOMINATION_INFO:
            info = config.DENOMINATION_INFO[confirmed_denom]
            details = {
                "symbol": info["symbol"],
                "name": info["name"],
                "motif_name": info["motif"],
                "base_color": info["base_color"]
            }

        return CurrencyResult(
            is_valid=is_valid,
            denomination=confirmed_denom,
            confidence=confirmed_conf,
            class_probs=class_probs,
            details=details
        )
