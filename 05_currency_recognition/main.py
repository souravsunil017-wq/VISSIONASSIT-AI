"""
Feature 5: Deep Learning Indian Currency Denomination Recognition System
Pre-Trained PyTorch MobileNetV3 Convolutional Neural Network trained on Kaggle Indian Currency Dataset.
"""

import sys
import os
import time
import argparse
import cv2
import numpy as np

# Add parent directory to path to allow importing shared modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared import vision_utils
from shared.camera_stream import ThreadedCameraStream, select_camera_source, normalize_ip_url

try:
    from . import config
    from .classifier_engine import MobileNetCurrencyClassifier, CurrencyResult
    from .currency_voice import CurrencyVoiceAssistant
except ImportError:
    import config
    from classifier_engine import MobileNetCurrencyClassifier, CurrencyResult
    from currency_voice import CurrencyVoiceAssistant

def draw_deep_currency_hud(frame: np.ndarray, result: CurrencyResult, voice_assistant: CurrencyVoiceAssistant, fps: float, source_label: str):
    """
    Renders an Explainable AI HUD overlay displaying Pre-Trained MobileNetV3 deep learning
    probabilities, detected banknote info, and live wallet tally.
    """
    h, w = frame.shape[:2]

    # Top Header
    vision_utils.draw_header_hud(frame, title="FEATURE 5: INR CURRENCY RECOGNITION", subtitle=f"Pretrained PyTorch MobileNetV3 | {source_label}", fps=fps, is_muted=voice_assistant.is_muted)

    # 1. Target Banknote Focus Guide Box
    cw, ch = int(w * 0.75), int(h * 0.70)
    cx, cy = int((w - cw) / 2), int((h - ch) / 2)

    box_color = (0, 230, 0) if result.is_valid_note else (70, 80, 90)
    cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), box_color, 2)

    if result.is_valid_note:
        sym = result.layer_details.get("symbol", f"₹{result.denomination}")
        tag = f"{sym} [{int(result.final_confidence * 100)}%]"
        cv2.rectangle(frame, (cx, max(0, cy - 28)), (cx + len(tag) * 13 + 10, cy), (0, 180, 0), -1)
        cv2.putText(frame, tag, (cx + 6, cy - 8), cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. Left Explainable AI (XAI) Deep Learning Metric Card
    card_x, card_y, card_w, card_h = 20, 80, 290, 215
    vision_utils.draw_transparent_rect(frame, card_x, card_y, card_w, card_h, (20, 25, 30), alpha=0.88, border_color=(80, 90, 100), border_thickness=1)

    cv2.putText(frame, "DEEP CNN PROBABILITIES", (card_x + 10, card_y + 22), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 200), 1, cv2.LINE_AA)
    cv2.line(frame, (card_x + 10, card_y + 28), (card_x + card_w - 10, card_y + 28), (100, 100, 110), 1)

    # Display probabilities for denominations
    probs = result.class_probs
    denoms_display = ["10", "20", "50", "100", "200", "500"]
    for idx, d in enumerate(denoms_display):
        p_val = probs.get(d, 0.0)
        p_pct = int(p_val * 100)
        ry = card_y + 48 + (idx * 22)
        
        # Color highlight if top prediction
        text_col = (0, 255, 120) if (result.is_valid_note and result.denomination == d) else (200, 200, 200)
        cv2.putText(frame, f"Rs. {d:>3}:", (card_x + 12, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.40, text_col, 1, cv2.LINE_AA)
        
        # Mini bar
        bar_len = int(p_val * 110)
        cv2.rectangle(frame, (card_x + 85, ry - 10), (card_x + 85 + bar_len, ry + 2), text_col, -1)
        cv2.putText(frame, f"{p_pct}%", (card_x + 205, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    bg_prob = int(probs.get("background", 0.0) * 100)
    cv2.line(frame, (card_x + 10, card_y + 185), (card_x + card_w - 10, card_y + 185), (100, 100, 110), 1)
    status_col = (0, 255, 120) if result.is_valid_note else (150, 150, 150)
    status_label = f"Match: Rs. {result.denomination} ({int(result.final_confidence*100)}%)" if result.is_valid_note else f"Scanning... (Bg: {bg_prob}%)"
    cv2.putText(frame, status_label, (card_x + 12, card_y + 203), cv2.FONT_HERSHEY_DUPLEX, 0.45, status_col, 1, cv2.LINE_AA)

    # 3. Wallet Total Status Card (Right side)
    wal_w, wal_h = 240, 70
    wal_x = w - wal_w - 20
    wal_y = 80
    vision_utils.draw_transparent_rect(frame, wal_x, wal_y, wal_w, wal_h, (25, 20, 30), alpha=0.85, border_color=(120, 80, 140), border_thickness=1)
    cv2.putText(frame, "WALLET TOTAL (CASH)", (wal_x + 10, wal_y + 22), cv2.FONT_HERSHEY_DUPLEX, 0.42, (220, 180, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Rs. {voice_assistant.wallet_total} INR", (wal_x + 10, wal_y + 54), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 180), 2, cv2.LINE_AA)

    # 4. Bottom Denomination Banner
    banner_h = 75
    banner_y = h - banner_h - 15
    banner_w = w - 40
    banner_x = 20

    if result.is_valid_note:
        sym = result.layer_details.get("symbol", f"Rs. {result.denomination}")
        banner_title = f"IDENTIFIED: {sym} BANKNOTE"
        banner_sub = f"Motif: {result.layer_details.get('motif_name', '')} | Color: {result.layer_details.get('base_color', '')} | Confidence: {int(result.final_confidence*100)}%"
        vision_utils.draw_transparent_rect(frame, banner_x, banner_y, banner_w, banner_h, (0, 140, 40), alpha=0.9, border_color=(255, 255, 255), border_thickness=2)
        cv2.putText(frame, banner_title, (banner_x + 20, banner_y + 35), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, banner_sub, (banner_x + 20, banner_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 255, 230), 1, cv2.LINE_AA)
    else:
        vision_utils.draw_transparent_rect(frame, banner_x, banner_y, banner_w, banner_h, (35, 35, 40), alpha=0.8, border_color=(100, 100, 110), border_thickness=1)
        cv2.putText(frame, "HOLD INDIAN BANKNOTE UP TO CAMERA", (banner_x + 20, banner_y + 35), cv2.FONT_HERSHEY_DUPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "Hotkeys: [A] Add to Wallet | [T] Speak Total | [R] Reset Wallet | [V] Mute Voice", (banner_x + 20, banner_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 170), 1, cv2.LINE_AA)

def run_currency_system(source=None, ask=False, stop_event=None):
    print("=" * 65)
    print("  FEATURE 5: PRE-TRAINED MOBILENETV3 CURRENCY RECOGNITION")
    print("=" * 65)

    if source is None:
        source, source_type = select_camera_source(ask=ask)
    else:
        if isinstance(source, str) and not source.isdigit():
            source = normalize_ip_url(source)
            source_type = "IP Camera"
        else:
            source = int(source)
            source_type = f"Webcam {source}"

    print(f"\n[Camera] Connecting to {source_type}: {source}")
    cap = ThreadedCameraStream(source)
    time.sleep(0.5)

    if not cap.isOpened() or not cap.read()[0]:
        print(f"[Error] Failed to connect to camera: {source}")
        print("[Tip] Check if IP Camera is active on phone or verify webcam index.")
        cap.release()
        return

    print("[Camera] Connected successfully!")
    print("\nControls:")
    print("  [A] - Add current recognized note to Wallet Total")
    print("  [T] - Speak Total Wallet Balance")
    print("  [R] - Reset Wallet")
    print("  [V] - Toggle Voice (Mute / Unmute)")
    print("  [Q] / [ESC] - Quit Application\n")

    classifier = MobileNetCurrencyClassifier()
    voice_assistant = CurrencyVoiceAssistant()
    voice_assistant.tts.speak("Currency recognition online. Pre trained deep vision model ready.", force=True)

    prev_time = time.time()
    fps = 0.0

    window_name = "Feature 5: Pre-Trained Deep Learning Currency Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break

            current_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(0.001, (current_time - prev_time)))
            prev_time = current_time

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

            # Run Pre-Trained Deep Neural Banknote Inference (12ms)
            result = classifier.predict(frame)

            # Announce recognized denomination aloud
            if result.is_valid_note:
                voice_assistant.announce_denomination(result)

            # Render Explainable AI HUD
            source_label = f"IP CAM: {str(source)[:20]}" if "http" in str(source) or ":" in str(source) else f"WEBCAM {source}"
            draw_deep_currency_hud(frame, result, voice_assistant, fps, source_label)

            cv2.imshow(window_name, frame)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key in [ord('v'), ord('V')]:
                voice_assistant.toggle_mute()
            elif key in [ord('a'), ord('A')]:
                if result.is_valid_note:
                    voice_assistant.announce_denomination(result, add_to_wallet=True)
            elif key in [ord('t'), ord('T')]:
                voice_assistant.speak_total()
            elif key in [ord('r'), ord('R')]:
                voice_assistant.reset_wallet()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        voice_assistant.shutdown()
        print("[App] Closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature 5: Pre-Trained Currency Recognition")
    parser.add_argument("--source", type=str, default=None, help="Webcam index or IP camera URL")
    parser.add_argument("--ip", type=str, default=None, help="IP Camera address")
    parser.add_argument("--ask", action="store_true", help="Prompt for camera choice on startup")
    args = parser.parse_args()

    target_source = args.ip if args.ip is not None else args.source
    if target_source is not None and str(target_source).isdigit():
        target_source = int(target_source)

    run_currency_system(source=target_source, ask=args.ask)
