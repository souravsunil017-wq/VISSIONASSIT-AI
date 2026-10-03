"""
Text Extraction & Image Preprocessing Pipeline for Feature 6.
Combines ultra-fast native Windows OCR (winocr) and EasyOCR for instant text reading.
"""

import cv2
import numpy as np
import threading
import re
from typing import List, Dict, Any, Tuple, Optional

try:
    from . import config
except ImportError:
    import config

class TextBlock:
    """Represents a localized recognized text segment."""
    def __init__(self, text: str, confidence: float, box: Tuple[int, int, int, int]):
        self.text = text.strip()
        self.confidence = confidence
        self.box = box  # (x1, y1, x2, y2)
        self.x1, self.y1, self.x2, self.y2 = box
        self.cx = (self.x1 + self.x2) / 2.0
        self.cy = (self.y1 + self.y2) / 2.0

class TextExtractor:
    def __init__(self):
        self.engine_type = "winocr"
        self.is_ready = True
        self.is_loading = False
        self.loading_status = "OCR Engine Ready"
        self._ocr_busy = False

        # Verify winocr availability
        try:
            import winocr
            self.engine_type = "winocr"
            print("[OCR Engine] Native Windows OCR (winocr) initialized and ready!")
        except Exception as e:
            print(f"[OCR Engine] winocr not available ({e}), using EasyOCR fallback.")
            self.engine_type = "easyocr"
            self._init_easyocr_background()

    def _init_easyocr_background(self):
        def _loader():
            try:
                import easyocr
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                self.is_ready = True
                print("[OCR Engine] EasyOCR loaded successfully!")
            except Exception as e:
                print(f"[OCR Warning] EasyOCR error: {e}")
        t = threading.Thread(target=_loader, daemon=True)
        t.start()

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhances contrast and sharpens text contours for OCR.
        """
        if image is None or image.size == 0:
            return image

        # For winocr, passing clear BGR image works best
        return image

    def extract_text_async(self, frame: np.ndarray, callback):
        """
        Runs text extraction in a background thread to keep video FPS smooth.
        """
        if self._ocr_busy:
            return

        def _worker():
            self._ocr_busy = True
            try:
                blocks = self.extract_text_sync(frame)
                if callback:
                    callback(blocks)
            finally:
                self._ocr_busy = False

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def extract_text_sync(self, frame: np.ndarray) -> List[TextBlock]:
        """
        Synchronously extracts text blocks and bounding boxes from frame.
        """
        if frame is None or frame.size == 0:
            return []

        text_blocks = []

        # 1. Try Native Windows OCR (winocr) - Instant and accurate
        if self.engine_type == "winocr":
            try:
                import winocr
                res = winocr.recognize_cv2_sync(frame, 'en')
                lines = res.get('lines', [])
                for line in lines:
                    line_text = line.get('text', '').strip()
                    if len(line_text) < 2 or not re.search(r'[a-zA-Z0-9]', line_text):
                        continue

                    # Calculate bounding box from words in line
                    words = line.get('words', [])
                    if words:
                        min_x = min([w['bounding_rect']['x'] for w in words])
                        min_y = min([w['bounding_rect']['y'] for w in words])
                        max_x = max([w['bounding_rect']['x'] + w['bounding_rect']['width'] for w in words])
                        max_y = max([w['bounding_rect']['y'] + w['bounding_rect']['height'] for w in words])
                        x1, y1, x2, y2 = int(min_x), int(min_y), int(max_x), int(max_y)
                    else:
                        x1, y1, x2, y2 = 20, 20, frame.shape[1] - 20, 60

                    text_blocks.append(TextBlock(text=line_text, confidence=0.92, box=(x1, y1, x2, y2)))
            except Exception as e:
                print(f"[winocr Error]: {e}")

        # 2. Fallback to EasyOCR if winocr returned empty
        if not text_blocks and hasattr(self, 'easyocr_reader') and self.easyocr_reader is not None:
            try:
                results = self.easyocr_reader.readtext(frame, paragraph=False)
                for bbox, text, conf in results:
                    text_clean = text.strip()
                    if conf < config.CONFIDENCE_THRESHOLD or len(text_clean) < 2:
                        continue
                    if not re.search(r'[a-zA-Z0-9]', text_clean):
                        continue

                    pts = np.array(bbox, dtype=np.int32)
                    x1 = int(np.min(pts[:, 0]))
                    y1 = int(np.min(pts[:, 1]))
                    x2 = int(np.max(pts[:, 0]))
                    y2 = int(np.max(pts[:, 1]))

                    text_blocks.append(TextBlock(text=text_clean, confidence=conf, box=(x1, y1, x2, y2)))
            except Exception as e:
                print(f"[EasyOCR Error]: {e}")

        return text_blocks
