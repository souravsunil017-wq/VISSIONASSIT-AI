"""
Feature 6: Text Reading & Audio Narration System
Main Application Runner with Instant Camera Window & Asynchronous OCR Pipeline.
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
    from .text_extractor import TextExtractor, TextBlock
    from .reader_voice import TextReadingVoiceAssistant
except ImportError:
    import config
    from text_extractor import TextExtractor, TextBlock
    from reader_voice import TextReadingVoiceAssistant

def draw_ocr_hud(frame: np.ndarray, text_blocks: list, voice: TextReadingVoiceAssistant, extractor: TextExtractor, fps: float, source_label: str, is_snapshot_frozen: bool):
    """
    Renders an augmented reality OCR HUD with text bounding boxes,
    live transcription preview, and narration progress cards.
    """
    h, w = frame.shape[:2]

    # 1. Top Header HUD
    mode_title = "DOCUMENT READING (FROZEN)" if is_snapshot_frozen else "LIVE SIGNBOARD SPOTTER"
    vision_utils.draw_header_hud(frame, title="FEATURE 6: TEXT READING (OCR)", subtitle=f"{mode_title} | {source_label}", fps=fps, is_muted=voice.is_muted)

    # Engine Status Pill (Top Right under FPS)
    if extractor.is_loading:
        status_txt = "OCR LOADING..."
        status_col = (0, 180, 255)
    elif extractor.is_ready:
        status_txt = "OCR READY"
        status_col = (0, 255, 120)
    else:
        status_txt = "OCR ACTIVE"
        status_col = (200, 200, 200)
    cv2.putText(frame, status_txt, (w - 180, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_col, 1, cv2.LINE_AA)

    # 2. Draw Bounding Boxes on Detected Text Lines
    for b in text_blocks:
        x1, y1, x2, y2 = b.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 120), 2)
        
        # Text label above box
        preview_text = b.text[:22] + "..." if len(b.text) > 22 else b.text
        tag_bg_w = max(60, len(preview_text) * 8 + 10)
        cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + tag_bg_w, y1), (0, 180, 80), -1)
        cv2.putText(frame, preview_text, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    # 3. Document Reading Mode Narration Card
    if is_snapshot_frozen or voice.is_document_mode:
        card_w, card_h = w - 40, 110
        card_x, card_y = 20, h - card_h - 15
        vision_utils.draw_transparent_rect(frame, card_x, card_y, card_w, card_h, (20, 25, 35), alpha=0.9, border_color=(0, 220, 255), border_thickness=2)

        tot = len(voice.document_sentences)
        cur = min(tot, voice.current_idx + 1)
        status_text = "PAUSED" if voice.is_paused else ("READING" if voice.is_document_mode else "READY")
        
        cv2.putText(frame, f"DOCUMENT NARRATOR [{status_text}] (Sentence {cur}/{tot})", (card_x + 15, card_y + 25), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 200), 1, cv2.LINE_AA)

        # Show current sentence
        curr_sentence = voice.document_sentences[voice.current_idx] if (voice.document_sentences and voice.current_idx < tot) else "Document processed."
        disp_sent = curr_sentence[:75] + "..." if len(curr_sentence) > 75 else curr_sentence
        cv2.putText(frame, f"\"{disp_sent}\"", (card_x + 15, card_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        # Controls banner
        cv2.putText(frame, "[SPACE] Unfreeze | [P] Pause/Resume | [R] Repeat | [N] Next | [V] Mute Voice", (card_x + 15, card_y + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 210, 255), 1, cv2.LINE_AA)

    else:
        # Live Spotter Bottom Banner
        banner_h = 75
        banner_y = h - banner_h - 15
        banner_w = w - 40
        banner_x = 20
        vision_utils.draw_transparent_rect(frame, banner_x, banner_y, banner_w, banner_h, (25, 30, 35), alpha=0.85, border_color=(80, 90, 100), border_thickness=1)

        if text_blocks:
            found_summary = " ".join([b.text for b in text_blocks[:3]])
            found_disp = found_summary[:65] + "..." if len(found_summary) > 65 else found_summary
            cv2.putText(frame, f"LIVE TEXT: \"{found_disp}\"", (banner_x + 15, banner_y + 32), cv2.FONT_HERSHEY_DUPLEX, 0.58, (0, 255, 180), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "POINT AT SIGNBOARD, PAGE, OR PACKAGING", (banner_x + 15, banner_y + 32), cv2.FONT_HERSHEY_DUPLEX, 0.58, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.putText(frame, "Press [SPACE] to Freeze & Read Full Document | [V] Mute | [Q] Quit", (banner_x + 15, banner_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 230), 1, cv2.LINE_AA)

def run_text_reading_system(source=None, ask=False, stop_event=None):
    print("=" * 65)
    print("  FEATURE 6: TEXT READING & AUDIO NARRATION SYSTEM (OCR)")
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
    print("  [SPACE] - Freeze & Read Full Document / Page")
    print("  [P]     - Pause / Resume Audio Narration")
    print("  [R]     - Repeat Last Sentence")
    print("  [N]     - Skip to Next Sentence")
    print("  [V]     - Toggle Voice Guidance (Mute / Unmute)")
    print("  [Q]/ESC - Quit Application\n")

    window_name = "Feature 6: Text Reading (OCR & Audio Voice)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    # Initialize extractor (loads in background without freezing video)
    extractor = TextExtractor()
    voice = TextReadingVoiceAssistant()
    voice.tts.speak("Text reading online.", force=True)

    is_frozen = False
    frozen_frame = None
    frozen_blocks = []

    last_live_ocr_time = 0.0
    live_blocks = []
    prev_time = time.time()
    fps = 0.0

    def _on_live_ocr_complete(blocks):
        nonlocal live_blocks
        live_blocks = blocks
        voice.announce_live_spot(blocks)

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break

            current_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(0.001, (current_time - prev_time)))
            prev_time = current_time

            if is_frozen and frozen_frame is not None:
                display_frame = frozen_frame.copy()
                active_blocks = frozen_blocks
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                display_frame = cv2.resize(frame, (640, 480))

                # Asynchronous live signboard spotter (never blocks camera loop)
                if (current_time - last_live_ocr_time) >= config.LIVE_SPOTTER_INTERVAL_SEC:
                    enhanced = extractor.preprocess_image(display_frame)
                    extractor.extract_text_async(enhanced, _on_live_ocr_complete)
                    last_live_ocr_time = current_time

                active_blocks = live_blocks

            # Render Augmented Reality OCR HUD
            source_label = f"IP CAM: {str(source)[:20]}" if "http" in str(source) or ":" in str(source) else f"WEBCAM {source}"
            draw_ocr_hud(display_frame, active_blocks, voice, extractor, fps, source_label, is_frozen)

            cv2.imshow(window_name, display_frame)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]: # Q or ESC
                break
            elif key == ord(' '): # SPACE: Freeze snapshot & Read Document
                if is_frozen:
                    print("[Mode] Returning to Live Signboard Spotter.")
                    is_frozen = False
                    frozen_frame = None
                    voice.stop_document_reading()
                else:
                    print("[Mode] Freezing snapshot & parsing document text...")
                    ret, raw_frame = cap.read()
                    if ret and raw_frame is not None:
                        is_frozen = True
                        frozen_frame = cv2.resize(raw_frame, (640, 480))
                        enhanced = extractor.preprocess_image(frozen_frame)
                        frozen_blocks = extractor.extract_text_sync(enhanced)
                        voice.start_document_reading(frozen_blocks)

            elif key in [ord('p'), ord('P')]: # Pause / Resume
                voice.pause_or_resume()
            elif key in [ord('r'), ord('R')]: # Repeat
                voice.repeat_sentence()
            elif key in [ord('n'), ord('N')]: # Next
                voice.next_sentence()
            elif key in [ord('v'), ord('V')]: # Toggle Voice Mute
                voice.toggle_mute()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        voice.shutdown()
        print("[App] Closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature 6: Text Reading for Visually Impaired")
    parser.add_argument("--source", type=str, default=None, help="Webcam index or IP camera URL")
    parser.add_argument("--ip", type=str, default=None, help="IP Camera address")
    parser.add_argument("--ask", action="store_true", help="Prompt for camera choice on startup")
    args = parser.parse_args()

    target_source = args.ip if args.ip is not None else args.source
    if target_source is not None and str(target_source).isdigit():
        target_source = int(target_source)

    run_text_reading_system(source=target_source, ask=args.ask)
