"""
Feature 2: Approaching Vehicle Detection & Time-to-Collision (TTC) Warning System.
Main Real-Time Video Stream Runner with Explainable AI (XAI) HUD and Voice Guidance.
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
    from .vehicle_detector import VehicleDetector
    from .motion_tracker import VehicleMotionTracker, VehicleTrack
    from .vehicle_voice import VehicleVoiceAssistant
except ImportError:
    import config
    from vehicle_detector import VehicleDetector
    from motion_tracker import VehicleMotionTracker, VehicleTrack
    from vehicle_voice import VehicleVoiceAssistant

def draw_vehicle_hud(frame: np.ndarray, tracks: list, voice_assistant: VehicleVoiceAssistant, fps: float, source_label: str):
    """
    Renders an Explainable AI HUD overlay displaying vehicle bounding boxes,
    optical expansion motion vectors, Time-to-Collision (TTC), and corridor zones.
    """
    h, w = frame.shape[:2]

    # Top Header
    vision_utils.draw_header_hud(frame, title="FEATURE 2: APPROACHING VEHICLE & TTC SYSTEM",
                                subtitle=f"YOLOv8 + Optical Expansion | {source_label}",
                                fps=fps, is_muted=voice_assistant.is_muted)

    # 1. Spatial Corridor Boundary Guides
    lx = int(w * config.CORRIDOR_LEFT_MAX)
    rx = int(w * config.CORRIDOR_CENTER_MAX)

    cv2.line(frame, (lx, 55), (lx, h - 85), (80, 80, 90), 1, cv2.LINE_AA)
    cv2.line(frame, (rx, 55), (rx, h - 85), (80, 80, 90), 1, cv2.LINE_AA)

    cv2.putText(frame, "LEFT", (lx - 55, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 150), 1, cv2.LINE_AA)
    cv2.putText(frame, "CENTER PATH", (lx + int((rx - lx) / 2) - 45, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "RIGHT", (rx + 15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 150), 1, cv2.LINE_AA)

    # 2. Render Tracked Vehicle Overlays
    highest_threat = "SAFE"
    critical_count = 0
    warning_count = 0

    for trk in tracks:
        x1, y1, x2, y2 = trk.current_box
        threat = trk.threat_level

        # Threat color scheme
        if threat == "CRITICAL":
            box_color = (0, 0, 255)       # Red
            tag_color = (0, 0, 220)
            status_text = f"DANGER: TTC {trk.ttc_seconds}s"
            highest_threat = "CRITICAL"
            critical_count += 1
        elif threat == "WARNING":
            box_color = (0, 165, 255)     # Orange
            tag_color = (0, 140, 230)
            status_text = f"WARN: TTC {trk.ttc_seconds}s"
            if highest_threat != "CRITICAL":
                highest_threat = "WARNING"
            warning_count += 1
        elif threat == "RECEDING":
            box_color = (255, 180, 0)     # Cyan-Blue
            tag_color = (200, 140, 0)
            status_text = "Moving Away"
        else:
            box_color = (0, 230, 0)       # Green
            tag_color = (0, 160, 0)
            status_text = "Stationary / Safe"

        # Draw vehicle bounding box & corner brackets
        vision_utils.draw_corner_brackets(frame, x1, y1, x2 - x1, y2 - y1, color=box_color, thickness=2, length=18)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1)

        # Label tag
        label = f"{trk.class_name.upper()} #{trk.track_id} | {trk.current_distance}m"
        tag_w = max(130, len(label) * 9 + 10)
        cv2.rectangle(frame, (x1, max(0, y1 - 38)), (x1 + tag_w, y1), tag_color, -1)
        cv2.putText(frame, label, (x1 + 6, y1 - 22), cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, status_text, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw optical expansion / velocity indicator arrow
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        if trk.expansion_rate > 0.05:
            # Approaching arrow (pointing downward/toward viewer)
            arrow_len = min(40, int(trk.expansion_rate * 150))
            cv2.arrowedLine(frame, (cx, cy), (cx, cy + arrow_len), (0, 0, 255), 2, tipLength=0.35)
        elif trk.expansion_rate < -0.05:
            # Receding arrow (pointing upward/away)
            arrow_len = min(40, int(abs(trk.expansion_rate) * 150))
            cv2.arrowedLine(frame, (cx, cy), (cx, cy - arrow_len), (255, 180, 0), 2, tipLength=0.35)

    # 3. Left Telemetry Threat Card
    card_x, card_y, card_w, card_h = 20, 80, 260, 160
    vision_utils.draw_transparent_rect(frame, card_x, card_y, card_w, card_h, (20, 25, 30), alpha=0.88, border_color=(80, 90, 100), border_thickness=1)
    
    cv2.putText(frame, "TRAFFIC MOTION TELEMETRY", (card_x + 10, card_y + 22), cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 255, 200), 1, cv2.LINE_AA)
    cv2.line(frame, (card_x + 10, card_y + 28), (card_x + card_w - 10, card_y + 28), (100, 100, 110), 1)

    cv2.putText(frame, f"Active Vehicles: {len(tracks)}", (card_x + 12, card_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    
    crit_col = (0, 0, 255) if critical_count > 0 else (180, 180, 180)
    cv2.putText(frame, f"Critical Hazards (TTC < 2.5s): {critical_count}", (card_x + 12, card_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.42, crit_col, 1, cv2.LINE_AA)
    
    warn_col = (0, 165, 255) if warning_count > 0 else (180, 180, 180)
    cv2.putText(frame, f"Warning Alerts (TTC < 5.0s): {warning_count}", (card_x + 12, card_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.42, warn_col, 1, cv2.LINE_AA)

    status_str = "SYSTEM STATUS: " + highest_threat
    stat_col = (0, 0, 255) if highest_threat == "CRITICAL" else ((0, 165, 255) if highest_threat == "WARNING" else (0, 255, 120))
    cv2.putText(frame, status_str, (card_x + 12, card_y + 145), cv2.FONT_HERSHEY_DUPLEX, 0.45, stat_col, 1, cv2.LINE_AA)

    # 4. Bottom Collision Risk Banner
    banner_h = 75
    banner_y = h - banner_h - 15
    banner_w = w - 40
    banner_x = 20

    if highest_threat == "CRITICAL":
        vision_utils.draw_transparent_rect(frame, banner_x, banner_y, banner_w, banner_h, (0, 0, 180), alpha=0.92, border_color=(0, 0, 255), border_thickness=2)
        cv2.putText(frame, "EMERGENCY: VEHICLE APPROACHING FAST!", (banner_x + 20, banner_y + 35), cv2.FONT_HERSHEY_DUPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Immediate collision risk detected in walking path. Stop or step aside.", (banner_x + 20, banner_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 255), 1, cv2.LINE_AA)
    elif highest_threat == "WARNING":
        vision_utils.draw_transparent_rect(frame, banner_x, banner_y, banner_w, banner_h, (0, 110, 180), alpha=0.90, border_color=(0, 165, 255), border_thickness=2)
        cv2.putText(frame, "CAUTION: VEHICLE APPROACHING", (banner_x + 20, banner_y + 35), cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Vehicle detected moving towards your trajectory. Maintain awareness.", (banner_x + 20, banner_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 240, 255), 1, cv2.LINE_AA)
    else:
        vision_utils.draw_transparent_rect(frame, banner_x, banner_y, banner_w, banner_h, (25, 30, 35), alpha=0.85, border_color=(80, 90, 100), border_thickness=1)
        cv2.putText(frame, "PATH CLEAR - TRAFFIC MONITOR ACTIVE", (banner_x + 20, banner_y + 35), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, "Monitoring road vehicles, optical expansion, and collision risks in real time.", (banner_x + 20, banner_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 170, 180), 1, cv2.LINE_AA)

def run_vehicle_system(source=None, ask=False, stop_event=None):
    print("=" * 65)
    print("  FEATURE 2: APPROACHING VEHICLE & TIME-TO-COLLISION (TTC) SYSTEM")
    print("=" * 65)

    if source is None:
        source, source_type = select_camera_source(ask=ask)
    else:
        if isinstance(source, str) and not source.isdigit():
            source = normalize_ip_url(source)
            source_type = "IP Camera"
        else:
            source = int(source)
            source_type = "Webcam"

    print(f"\n[Camera] Connecting to {source_type}: {source}")
    cap = ThreadedCameraStream(source)
    time.sleep(0.5)

    if not cap.isOpened() or not cap.read()[0]:
        print(f"[Error] Failed to connect to camera: {source}")
        cap.release()
        return

    print("[Camera] Connected successfully!")
    print("\nControls:")
    print("  [V] - Toggle Voice Announcements (Mute / Unmute)")
    print("  [Q] / [ESC] - Quit Application\n")

    detector = VehicleDetector()
    tracker = VehicleMotionTracker()
    voice_assistant = VehicleVoiceAssistant()
    voice_assistant.tts.speak("Approaching vehicle collision warning system online.", force=True)

    prev_time = time.time()
    fps = 0.0

    window_name = "Feature 2: Approaching Vehicle & Time-to-Collision (TTC) Detection"
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

            # 1. Detect all vehicles
            detections = detector.detect_vehicles(frame)

            # 2. Track motion & calculate Optical Expansion and TTC
            tracks = tracker.update_tracks(detections)

            # 3. Audio Voice Guidance
            voice_assistant.evaluate_and_announce_threats(tracks)

            # 4. Render Explainable AI HUD
            source_label = f"IP CAM: {str(source)[:20]}" if "http" in str(source) or ":" in str(source) else f"WEBCAM {source}"
            draw_vehicle_hud(frame, tracks, voice_assistant, fps, source_label)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key in [ord('v'), ord('V')]:
                voice_assistant.toggle_mute()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        voice_assistant.shutdown()
        print("[Vehicle System] Closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature 2: Approaching Vehicle Detection")
    parser.add_argument("--source", type=str, default=None, help="Webcam index or IP camera URL")
    parser.add_argument("--ip", type=str, default=None, help="IP Camera address")
    parser.add_argument("--ask", action="store_true", help="Prompt for camera choice on startup")
    args = parser.parse_args()

    target_source = args.ip if args.ip is not None else args.source
    if target_source is not None and str(target_source).isdigit():
        target_source = int(target_source)

    run_vehicle_system(source=target_source, ask=args.ask)
