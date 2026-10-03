"""
Feature 3: Obstacle & Path-Hazard Detection System.
Real-time Video Stream Runner with Explainable AI (XAI) Multi-Zone HUD and Voice Guidance.
Detects potholes, ground drop-offs, stairs up/down, low-hanging overhead hazards, and slippery floors.
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
    from .pothole_detector import PotholeDetector, PotholeHazard
    from .staircase_detector import StaircaseDetector, StaircaseHazard
    from .overhead_detector import OverheadAndGlintDetector, OverheadHazard, WetSurfaceHazard
    from .hazard_voice import HazardVoiceAssistant
except ImportError:
    import config
    from pothole_detector import PotholeDetector, PotholeHazard
    from staircase_detector import StaircaseDetector, StaircaseHazard
    from overhead_detector import OverheadAndGlintDetector, OverheadHazard, WetSurfaceHazard
    from hazard_voice import HazardVoiceAssistant

def draw_hazard_hud(frame: np.ndarray,
                    potholes: list,
                    stair: StaircaseHazard,
                    overhead: OverheadHazard,
                    wet_surface: WetSurfaceHazard,
                    voice_assistant: HazardVoiceAssistant,
                    fps: float,
                    source_label: str,
                    show_zones: bool = True):
    """
    Renders an Explainable AI HUD overlay displaying multi-zone boundary guides,
    pothole cavity bounding boxes, staircase horizontal step risers,
    overhead hazard box, wet glint indicators, and hazard telemetry.
    """
    h, w = frame.shape[:2]

    # 1. Top Header
    vision_utils.draw_header_hud(
        frame,
        title="FEATURE 3: OBSTACLE & PATH HAZARD DETECTION",
        subtitle=f"Multi-Zone Spatial CV | {source_label}",
        fps=fps,
        is_muted=voice_assistant.is_muted
    )

    # 2. Multi-Zone Spatial Corridor Lines
    lx = int(w * config.CORRIDOR_LEFT_MAX)
    rx = int(w * config.CORRIDOR_CENTER_MAX)
    cv2.line(frame, (lx, 55), (lx, h - 85), (60, 60, 70), 1, cv2.LINE_AA)
    cv2.line(frame, (rx, 55), (rx, h - 85), (60, 60, 70), 1, cv2.LINE_AA)

    # Zone Dividers (Overhead, Mid-Stairs, Floor Plane)
    if show_zones:
        y_overhead = int(h * config.OVERHEAD_ZONE_MAX_Y)
        y_ground = int(h * config.GROUND_FLOOR_ZONE_MIN_Y)
        cv2.line(frame, (10, y_overhead), (w - 10, y_overhead), (45, 45, 55), 1, cv2.LINE_AA)
        cv2.line(frame, (10, y_ground), (w - 10, y_ground), (45, 45, 55), 1, cv2.LINE_AA)
        cv2.putText(frame, "OVERHEAD ZONE", (15, y_overhead - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 140), 1, cv2.LINE_AA)
        cv2.putText(frame, "FLOOR & CAVITY PLANE", (15, y_ground + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 140), 1, cv2.LINE_AA)

    active_hazards_count = 0
    highest_severity = "CLEAR"

    # 3. Render Overhead Hazard (Purple / Magenta Alert Box)
    if overhead is not None:
        ox1, oy1, ox2, oy2 = overhead.box_xyxy
        box_color = (180, 0, 255)  # Magenta
        vision_utils.draw_corner_brackets(frame, ox1, oy1, ox2 - ox1, oy2 - oy1, color=box_color, thickness=2, length=18)
        cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), box_color, 1)

        tag_w = 260
        cv2.rectangle(frame, (ox1, max(0, oy1 - 32)), (ox1 + tag_w, oy1), (150, 0, 220), -1)
        cv2.putText(frame, f"HEAD HAZARD: {overhead.hazard_name}", (ox1 + 6, oy1 - 18), cv2.FONT_HERSHEY_DUPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Edge Density: {overhead.edge_density*100:.1f}% | DUCK HEAD", (ox1 + 6, oy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        
        active_hazards_count += 1
        highest_severity = "CRITICAL"

    # 4. Render Staircase Step Risers (Cyan / Amber Alert Lines)
    if stair is not None:
        stair_color = (0, 230, 255) if stair.stair_type == "Stairs Up" else (0, 90, 255)
        # Draw individual detected riser lines
        for lx1, ly1, lx2, ly2 in stair.lines:
            cv2.line(frame, (lx1, ly1), (lx2, ly2), stair_color, 2, cv2.LINE_AA)

        # Draw enclosing bounding box
        sx1, sy1, sx2, sy2 = stair.box_xyxy
        vision_utils.draw_corner_brackets(frame, sx1, sy1, sx2 - sx1, sy2 - sy1, color=stair_color, thickness=2, length=16)
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), stair_color, 1)

        tag_w = 230
        cv2.rectangle(frame, (sx1, max(0, sy1 - 32)), (sx1 + tag_w, sy1), (0, 140, 200) if stair.stair_type == "Stairs Up" else (0, 50, 200), -1)
        cv2.putText(frame, f"{stair.stair_type.upper()} ({stair.line_count} Risers)", (sx1 + 6, sy1 - 18), cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Distance: {stair.distance_meters}m | Conf: {int(stair.confidence*100)}%", (sx1 + 6, sy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        active_hazards_count += 1
        if highest_severity != "CRITICAL":
            highest_severity = "WARNING"

    # 5. Render Pothole & Floor Cavity Hazards (Red / Orange Alert Box)
    for p in potholes:
        px1, py1, px2, py2 = p.box_xyxy
        pot_color = (0, 0, 255) if p.distance_meters < 2.5 else (0, 165, 255)

        vision_utils.draw_corner_brackets(frame, px1, py1, px2 - px1, py2 - py1, color=pot_color, thickness=2, length=14)
        cv2.rectangle(frame, (px1, py1), (px2, py2), pot_color, 1)

        tag_w = 190
        cv2.rectangle(frame, (px1, max(0, py1 - 30)), (px1 + tag_w, py1), (0, 0, 200) if p.distance_meters < 2.5 else (0, 130, 220), -1)
        cv2.putText(frame, f"{p.hazard_type.upper()}", (px1 + 6, py1 - 16), cv2.FONT_HERSHEY_DUPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{p.distance_meters}m | {p.corridor} | Conf: {int(p.confidence*100)}%", (px1 + 6, py1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

        active_hazards_count += 1
        if p.distance_meters < 2.0:
            highest_severity = "CRITICAL"
        elif highest_severity == "CLEAR":
            highest_severity = "WARNING"

    # 6. Render Wet / Slick Floor Glint Box (Blue / Teal Alert)
    if wet_surface is not None:
        wx1, wy1, wx2, wy2 = wet_surface.box_xyxy
        wet_color = (255, 200, 0)
        vision_utils.draw_corner_brackets(frame, wx1, wy1, wx2 - wx1, wy2 - wy1, color=wet_color, thickness=1, length=12)
        
        tag_w = 175
        cv2.rectangle(frame, (wx1, max(0, wy1 - 22)), (wx1 + tag_w, wy1), (180, 130, 0), -1)
        cv2.putText(frame, "SLICK / WET FLOOR", (wx1 + 4, wy1 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        active_hazards_count += 1
        if highest_severity == "CLEAR":
            highest_severity = "CAUTION"

    # 7. Bottom Telemetry Status Card
    bar_y1 = h - 75
    bar_y2 = h - 10
    bar_w = w - 20
    vision_utils.draw_transparent_rect(frame, 10, bar_y1, bar_w, bar_y2 - bar_y1, color=(15, 15, 20), alpha=0.85)
    cv2.rectangle(frame, (10, bar_y1), (10 + bar_w, bar_y2), (50, 50, 60), 1)

    # Threat Status Pill
    if highest_severity == "CRITICAL":
        pill_color = (0, 0, 220)
        status_msg = "CRITICAL PATH HAZARD DETECTED"
    elif highest_severity == "WARNING":
        pill_color = (0, 140, 230)
        status_msg = "PATH HAZARDS DETECTED - PROCEED WITH CAUTION"
    elif highest_severity == "CAUTION":
        pill_color = (200, 140, 0)
        status_msg = "SURFACE ANOMALIES DETECTED"
    else:
        pill_color = (0, 160, 0)
        status_msg = "PATH CLEAR - NO IMMEDIATE HAZARDS"

    cv2.rectangle(frame, (20, bar_y1 + 8), (140, bar_y1 + 32), pill_color, -1)
    cv2.putText(frame, highest_severity, (30, bar_y1 + 25), cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, status_msg, (155, bar_y1 + 25), cv2.FONT_HERSHEY_DUPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)

    # Sub-telemetry & Shortcuts
    telemetry_str = f"Active Hazards: {active_hazards_count} | Potholes: {len(potholes)} | Stairs: {'YES' if stair else 'NO'} | Overhead: {'YES' if overhead else 'NO'}"
    cv2.putText(frame, telemetry_str, (20, bar_y1 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 190), 1, cv2.LINE_AA)
    cv2.putText(frame, "[M] Mute  [Z] Toggle Zones  [C] Camera  [Q] Quit", (w - 320, bar_y1 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 150), 1, cv2.LINE_AA)

def run_hazard_system(source=None, ask: bool = False, stop_event=None):
    """
    Main execution loop for Feature 3.
    """
    print("\n=======================================================")
    print("  FEATURE 3: OBSTACLE & PATH HAZARD DETECTION SYSTEM")
    print("=======================================================")
    print("[INIT] Initializing multi-zone spatial hazard detectors...")

    pothole_det = PotholeDetector()
    stair_det = StaircaseDetector()
    overhead_det = OverheadAndGlintDetector()
    voice_assistant = HazardVoiceAssistant()

    # Camera selection
    if source is None:
        source_val, source_label = select_camera_source(ask=ask)
    else:
        if isinstance(source, str) and not source.isdigit():
            source_val = normalize_ip_url(source)
            source_label = "IP Camera"
        else:
            source_val = int(source)
            source_label = "Webcam"

    print(f"[CAMERA] Connecting to video feed ({source_label}): {source_val}...")
    stream = ThreadedCameraStream(source_val)
    time.sleep(0.8)

    if not stream.isOpened() or not stream.read()[0]:
        print(f"[ERROR] Failed to open video source: {source_val}")
        print("[HINT] If IP Camera is disconnected, pass '--source 0' to use the local webcam, or '--ask' to choose.")
        stream.release()
        return

    print("[SYSTEM] Path Hazard Detector Active. Press 'q' on video window to quit.")
    print("[AUDIO] Audio Guidance Online. Press 'm' to mute/unmute.")
    voice_assistant.tts.speak("Obstacle and path hazard detection system online.", force=True)

    fps = 0.0
    frame_count = 0
    start_time = time.time()
    show_zones = True

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                print("[SYSTEM] Stop event received from Master Hub.")
                break

            ret, frame = stream.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Resize to standard calibrated resolution
            frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
            display_frame = frame.copy()

            # Run parallel zone detectors
            potholes = pothole_det.detect_potholes(frame)
            stair = stair_det.detect_staircase(frame)
            overhead = overhead_det.detect_overhead_hazard(frame)
            wet_surface = overhead_det.detect_wet_surface(frame)

            # Evaluate hazards and trigger non-blocking voice guidance
            voice_assistant.evaluate_and_announce_hazards(
                potholes=potholes,
                stair=stair,
                overhead=overhead,
                wet_surface=wet_surface
            )

            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = round(frame_count / elapsed, 1)
                frame_count = 0
                start_time = time.time()

            # Render Explainable AI HUD Overlay
            draw_hazard_hud(
                frame=display_frame,
                potholes=potholes,
                stair=stair,
                overhead=overhead,
                wet_surface=wet_surface,
                voice_assistant=voice_assistant,
                fps=fps,
                source_label=source_label,
                show_zones=show_zones
            )

            cv2.imshow("Feature 3 - Path Hazard Detection", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[SYSTEM] Exiting Path Hazard Detection...")
                break
            elif key == ord('m') or key == ord('M'):
                is_muted = voice_assistant.toggle_mute()
                print(f"[AUDIO] Audio Muted: {is_muted}")
            elif key == ord('z') or key == ord('Z'):
                show_zones = not show_zones
                print(f"[HUD] Multi-Zone Guide Overlay: {show_zones}")
            elif key == ord('c') or key == ord('C'):
                print("[CAMERA] Switching camera stream...")
                stream.release()
                cv2.destroyAllWindows()
                source_val, source_label = select_camera_source(ask=True)
                stream = ThreadedCameraStream(source_val)
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[SYSTEM] Interrupted by user.")
    finally:
        stream.release()
        voice_assistant.shutdown()
        cv2.destroyAllWindows()
        print("[SYSTEM] Feature 3 shutdown cleanly.")

def run(source=None, ask: bool = False, stop_event=None):
    run_hazard_system(source=source, ask=ask, stop_event=stop_event)

def main():
    parser = argparse.ArgumentParser(description="Feature 3: Obstacle & Path-Hazard Detection System")
    parser.add_argument("--source", type=str, default=None, help="Camera index (0/1) or IP camera stream URL")
    parser.add_argument("--ip", type=str, default=None, help="Phone IP camera stream URL")
    parser.add_argument("--ask", action="store_true", help="Prompt for camera source on startup")
    args = parser.parse_args()

    input_source = args.ip if args.ip is not None else args.source
    run(source=input_source, ask=args.ask)

if __name__ == "__main__":
    main()
