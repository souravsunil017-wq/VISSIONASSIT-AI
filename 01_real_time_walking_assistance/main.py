"""
Feature 1: Real-time Walking Assistance
Real-time navigation and obstacle avoidance for visually impaired users.
"""

import sys
import os
import time
import argparse
import cv2

# Add parent directory to path to allow importing shared modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared import vision_utils
from shared.camera_stream import ThreadedCameraStream, select_camera_source
from shared.object_detector import UniversalObjectDetector

try:
    from . import config
    from .path_guidance import PathAnalyzer
    from .voice_feedback import WalkingVoiceAssistant
except ImportError:
    import config
    from path_guidance import PathAnalyzer
    from voice_feedback import WalkingVoiceAssistant

def run_walking_assistant(source=None, ask=False, stop_event=None):
    print("=" * 60)
    print("  AI ASSISTIVE SYSTEM - REAL-TIME WALKING ASSISTANCE")
    print("=" * 60)

    # Select camera (IP Camera or Webcam)
    if source is None:
        source, source_type = select_camera_source(ask=ask)
    else:
        source_type = "Webcam" if str(source).isdigit() else "IP Camera"

    print(f"\n[Camera] Connecting to {source_type}: {source}")
    cap = ThreadedCameraStream(source)
    time.sleep(0.5)

    if not cap.isOpened() or not cap.read()[0]:
        print(f"[Error] Failed to connect to camera: {source}")
        print("[Tip] Check if the IP Camera is running on your phone or check webcam connection.")
        cap.release()
        return

    print(f"[Camera] Connected successfully!")
    print("\nControls:")
    print("  [V] - Toggle Voice (Mute / Unmute)")
    print("  [D] - Toggle Zone Grid")
    print("  [Q] - Quit\n")

    # Initialize AI object detector, path guidance analyzer, and voice feedback
    detector = UniversalObjectDetector(conf_threshold=config.CONFIDENCE_THRESHOLD)
    path_analyzer = PathAnalyzer()
    voice_assistant = WalkingVoiceAssistant()
    voice_assistant.tts.speak("Walking assistance online.", force=True)

    show_debug_grid = True
    prev_time = time.time()
    fps = 0.0

    window_name = "Feature 1: Real-time Walking Assistance"
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

            # 1. Run Object Detection
            raw_detections = detector.detect(frame)

            # 2. Analyze Walking Corridors (Left, Center, Right)
            guidance, obstacles = path_analyzer.analyze_frame(raw_detections, frame.shape[1], frame.shape[0])

            # 3. Speak Guidance Instruction (Non-blocking)
            voice_assistant.process_guidance(guidance)

            # 4. Draw Augmented Reality Visual Overlay
            if show_debug_grid:
                vision_utils.draw_navigation_zones(frame, config.ZONE_LEFT_LIMIT, config.ZONE_RIGHT_LIMIT, config.GROUND_HORIZON_RATIO)

            for obs in obstacles:
                x1, y1, x2, y2 = obs['box']
                cls_name = obs['class_name']
                conf = obs['conf']
                dist_m = obs.get('distance_m', 0.0)
                zone = obs.get('zone', 'CENTER')
                is_close = obs.get('is_close', False)
                is_proximate = obs.get('is_proximate', False)

                if is_close:
                    color = (0, 0, 255)       # Red for immediate stop
                elif is_proximate:
                    color = (0, 165, 255)     # Orange for steering range
                else:
                    color = (255, 180, 0)     # Cyan/Blue for far / safe

                vision_utils.draw_corner_brackets(frame, x1, y1, x2 - x1, y2 - y1, color=color, thickness=2, length=14)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

                tag = f"{cls_name.upper()} [{zone}] {dist_m}m ({int(conf*100)}%)"
                tag_w = max(130, len(tag) * 8 + 10)
                cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + tag_w, y1), color, -1)
                cv2.putText(frame, tag, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

            source_label = f"IP CAM: {str(source)[:25]}" if "http" in str(source) or ":" in str(source) else f"WEBCAM {source}"
            vision_utils.draw_header_hud(frame, title="WALKING ASSISTANCE", subtitle=source_label, fps=fps, is_muted=voice_assistant.is_muted)
            vision_utils.draw_guidance_banner(frame, command_text=guidance.command, direction_code=guidance.direction_code, reason_text=guidance.reason)

            # 5. Display Result
            cv2.imshow(window_name, frame)

            # Key controls
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key in [ord('v'), ord('V')]:
                voice_assistant.toggle_mute()
            elif key in [ord('d'), ord('D')]:
                show_debug_grid = not show_debug_grid

    finally:
        cap.release()
        cv2.destroyAllWindows()
        voice_assistant.shutdown()
        print("[App] Closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature 1: Real-time Walking Assistance")
    parser.add_argument("--source", type=str, default=None, help="Webcam index (0) or IP Camera URL")
    parser.add_argument("--ip", type=str, default=None, help="IP Camera address (e.g. 192.168.1.5:8080)")
    parser.add_argument("--ask", action="store_true", help="Prompt for camera choice on startup")
    args = parser.parse_args()

    target_source = args.ip if args.ip is not None else args.source
    if target_source is not None and target_source.isdigit():
        target_source = int(target_source)

    run_walking_assistant(source=target_source, ask=args.ask)
