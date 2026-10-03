
"""
Feature 4: Familiar Person & Face Recognition System
Real-time face detection, deep embedding recognition, and spatial proximity audio guidance.
"""

import sys
import os
import time
import argparse
import cv2
import numpy as np

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared import vision_utils
from shared.camera_stream import ThreadedCameraStream, select_camera_source, normalize_ip_url

try:
    from . import config
    from .face_engine import FaceEngine, DetectedFace
    from .face_database import FaceDatabase
    from .face_voice import FaceVoiceAssistant
except ImportError:
    import config
    from face_engine import FaceEngine, DetectedFace
    from face_database import FaceDatabase
    from face_voice import FaceVoiceAssistant

def draw_face_hud(frame: np.ndarray, faces: list, enrolled_names: list, fps: float, source_label: str, is_muted: bool):
    """
    Renders augmented reality visual HUD overlay for face recognition.
    """
    h, w = frame.shape[:2]

    # 1. Top Header HUD
    subtitle = f"Enrolled: {len(enrolled_names)} Persons | {source_label}"
    vision_utils.draw_header_hud(frame, title="FEATURE 4: FAMILIAR PERSON RECOGNITION", subtitle=subtitle, fps=fps, is_muted=is_muted)

    # 2. Draw Navigation Corridor Guidelines (Subtle)
    x_left = int(w * config.ZONE_LEFT_LIMIT)
    x_right = int(w * config.ZONE_RIGHT_LIMIT)
    cv2.line(frame, (x_left, 60), (x_left, h - 80), (70, 75, 80), 1, cv2.LINE_AA)
    cv2.line(frame, (x_right, 60), (x_right, h - 80), (70, 75, 80), 1, cv2.LINE_AA)

    # 3. Render Face Detections
    for face in faces:
        x, y, fw, fh = face.box
        is_known = face.identity != "Unknown"
        
        box_color = (0, 230, 0) if is_known else (0, 165, 255)
        
        # Bounding Box
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), box_color, 2)

        # Draw 5 Facial Landmark Points (Right eye, Left eye, Nose, Mouth corners)
        if face.landmarks is not None:
            for pt in face.landmarks:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)

        # Name & Metric Tag
        if is_known:
            tag = f"{face.identity.upper()} [{int(face.match_score * 100)}%] ({face.distance_meters}m)"
            tag_color = (0, 180, 0)
        else:
            tag = f"UNKNOWN ({face.distance_meters}m)"
            tag_color = (0, 140, 230)

        tag_w = len(tag) * 10 + 10
        cv2.rectangle(frame, (x, max(0, y - 26)), (x + tag_w, y), tag_color, -1)
        cv2.putText(frame, tag, (x + 5, y - 7), cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # Spatial Zone Marker under box
        zone_label = f"[{face.zone}]"
        cv2.putText(frame, zone_label, (x + 5, y + fh + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

    # 4. Bottom Status Card
    banner_h = 75
    banner_y = h - banner_h - 10
    banner_w = w - 40
    banner_x = 20
    vision_utils.draw_transparent_rect(frame, banner_x, banner_y, banner_w, banner_h, (20, 25, 30), alpha=0.88, border_color=(255, 255, 255), border_thickness=1)

    if faces:
        known_in_frame = [f.identity for f in faces if f.identity != "Unknown"]
        if known_in_frame:
            summary = "DETECTED: " + ", ".join(set(known_in_frame))
            summary_color = (0, 255, 120)
        else:
            summary = f"DETECTED: {len(faces)} Unknown Person(s)"
            summary_color = (0, 200, 255)
        cv2.putText(frame, summary, (banner_x + 15, banner_y + 32), cv2.FONT_HERSHEY_DUPLEX, 0.58, summary_color, 1, cv2.LINE_AA)
    else:
        enrolled_summary = f"Enrolled: {', '.join(enrolled_names)}" if enrolled_names else "No photos enrolled yet"
        cv2.putText(frame, enrolled_summary[:60], (banner_x + 15, banner_y + 32), cv2.FONT_HERSHEY_DUPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.putText(frame, "Controls: [R] Reload Face Photos | [V] Mute Voice | [Q] Quit", (banner_x + 15, banner_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 180, 200), 1, cv2.LINE_AA)

def run_face_recognition_system(source=None, ask=False, stop_event=None):
    print("=" * 65)
    print("  FEATURE 4: FAMILIAR PERSON & FACE RECOGNITION SYSTEM")
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
        print("[Tip] Check if the IP Camera is running on your phone or check webcam connection.")
        cap.release()
        return

    print("[Camera] Connected successfully!")
    print("\nControls:")
    print("  [R] - Reload / Re-scan 'known_faces/' folder")
    print("  [V] - Toggle Voice (Mute / Unmute)")
    print("  [Q]/ESC - Quit Application\n")

    window_name = "Feature 4: Familiar Person & Face Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    # Initialize Engine, Database, and Voice Assistant
    engine = FaceEngine(input_size=(640, 480))
    db = FaceDatabase(engine=engine)
    voice = FaceVoiceAssistant()

    enrolled = db.get_enrolled_names()
    voice.speak_enrolled_summary(enrolled)

    prev_time = time.time()
    fps = 0.0

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

            # 1. Detect faces & extract 128-D vectors
            detected_faces = engine.detect_and_extract_faces(frame)

            # 2. Match against known identities
            for face in detected_faces:
                if face.feature_vector is not None:
                    name, score = db.identify_face(face.feature_vector)
                    face.identity = name
                    face.match_score = score

            # 3. Spatial Positional Voice Guidance
            voice.process_detected_faces(detected_faces)

            # 4. Render AR Visual HUD
            source_label = f"IP CAM: {str(source)[:20]}" if "http" in str(source) or ":" in str(source) else f"WEBCAM {source}"
            draw_face_hud(frame, detected_faces, db.get_enrolled_names(), fps, source_label, voice.is_muted)

            cv2.imshow(window_name, frame)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key in [ord('v'), ord('V')]:
                voice.toggle_mute()
            elif key in [ord('r'), ord('R')]:
                print("[Face Database] Rescanning known_faces directory...")
                db.rebuild_database()
                voice.speak_enrolled_summary(db.get_enrolled_names())

    finally:
        cap.release()
        cv2.destroyAllWindows()
        voice.shutdown()
        print("[App] Closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature 4: Familiar Person Recognition")
    parser.add_argument("--source", type=str, default=None, help="Webcam index or IP camera URL")
    parser.add_argument("--ip", type=str, default=None, help="IP Camera address")
    parser.add_argument("--ask", action="store_true", help="Prompt for camera choice on startup")
    args = parser.parse_args()

    target_source = args.ip if args.ip is not None else args.source
    if target_source is not None and str(target_source).isdigit():
        target_source = int(target_source)

    run_face_recognition_system(source=target_source, ask=args.ask)
