"""
Interactive Banknote Photo Capture & Auto-Enrollment Tool.
Allows the user to hold physical banknotes in front of the camera, capture photos,
and automatically update the recognition vector database.
"""

import os
import sys
import time
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.camera_stream import ThreadedCameraStream, select_camera_source, normalize_ip_url
from shared import vision_utils

try:
    from . import config
    from .currency_database import CurrencyDatabase
except ImportError:
    import config
    from currency_database import CurrencyDatabase

def run_capture_tool(source=None, ask=False):
    print("=" * 65)
    print("      BANKNOTE PHOTO CAPTURE & ENROLLMENT TOOL")
    print("=" * 65)
    print("Instructions:")
    print("  1. Select a denomination to capture (10, 20, 50, 100, 200, 500)")
    print("  2. Hold your physical banknote inside the green box on screen")
    print("  3. Press [SPACE] to capture photo (front & back recommended)")
    print("  4. Press [S] to Save and Auto-Enroll database")
    print("  5. Press [Q] to Exit")
    print("=" * 65 + "\n")

    if source is None:
        if ask:
            source, source_type = select_camera_source(ask=True)
        else:
            source = 0
            source_type = "Laptop Camera (Index 0)"
    else:
        if isinstance(source, str) and not source.isdigit():
            source = normalize_ip_url(source)
            source_type = "IP Camera"
        else:
            source = int(source)
            source_type = f"Webcam {source}"

    print(f"[Camera] Connecting to {source_type}: {source}")
    cap = ThreadedCameraStream(source)
    time.sleep(0.5)

    if not cap.isOpened() or not cap.read()[0]:
        print(f"[Error] Failed to connect to camera: {source}")
        cap.release()
        return

    current_denom_idx = 5  # Default to 500
    denoms = config.DENOMINATIONS
    captured_count = 0

    window_name = "Banknote Photo Capture Tool (Hold Note & Press SPACE)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
            h, w = frame.shape[:2]

            current_denom = denoms[current_denom_idx]
            folder_path = os.path.join(config.CURRENCY_NOTES_DIR, current_denom)
            os.makedirs(folder_path, exist_ok=True)
            existing_imgs = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

            # Banknote guide box
            cw, ch = int(w * 0.75), int(h * 0.70)
            cx, cy = int((w - cw) / 2), int((h - ch) / 2)
            cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2)

            # Header
            vision_utils.draw_header_hud(frame, title="BANKNOTE ENROLLMENT CAPTURE", subtitle=f"Target: Rs. {current_denom} | Photos: {len(existing_imgs)}", fps=30.0, is_muted=False)

            # Instructions Card
            vision_utils.draw_transparent_rect(frame, 20, h - 110, w - 40, 90, (20, 25, 30), alpha=0.85, border_color=(0, 255, 180), border_thickness=1)
            cv2.putText(frame, f"CURRENT DENOMINATION: Rs. {current_denom} ({config.DENOMINATION_INFO.get(current_denom, {}).get('name', '')})", (35, h - 85), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, "[SPACE]: Capture Photo  |  [N]: Next Denom  |  [P]: Prev Denom", (35, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "[S]: Save & Build Database Cache  |  [Q]: Quit", (35, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1, cv2.LINE_AA)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key == ord(' '):
                # Save cropped ROI as photo
                roi = frame[cy:cy+ch, cx:cx+cw]
                timestamp = int(time.time() * 1000)
                save_filename = f"note_{current_denom}_{timestamp}.jpg"
                save_path = os.path.join(folder_path, save_filename)
                cv2.imwrite(save_path, roi)
                captured_count += 1
                print(f"[Captured] Saved photo to: {save_path}")
                # Flash screen
                cv2.imshow(window_name, np.full_like(frame, 255))
                cv2.waitKey(80)
            elif key in [ord('n'), ord('N')]:
                current_denom_idx = (current_denom_idx + 1) % len(denoms)
            elif key in [ord('p'), ord('P')]:
                current_denom_idx = (current_denom_idx - 1) % len(denoms)
            elif key in [ord('s'), ord('S')]:
                print("\n[Database] Building vector cache from captured photos...")
                db = CurrencyDatabase()
                db.rebuild_database()
                print("[Database] Vector cache updated successfully!\n")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[Capture Tool] Finished.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Banknote Photo Capture Tool")
    parser.add_argument("--source", type=str, default=None, help="Webcam index or IP camera URL")
    parser.add_argument("--ip", type=str, default=None, help="IP Camera address")
    parser.add_argument("--ask", action="store_true", help="Prompt for camera choice on startup")
    args = parser.parse_args()

    target_source = args.ip if args.ip is not None else args.source
    if target_source is not None and str(target_source).isdigit():
        target_source = int(target_source)

    run_capture_tool(source=target_source, ask=args.ask)
