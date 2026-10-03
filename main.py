"""
AI Assistive Vision System for Visually Impaired
Master Application & Hands-Free Voice Command Hub.
"""

import sys
import os
import time
import threading
import argparse
import importlib
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from shared import vision_utils
from shared.audio_engine import TextToSpeechEngine
from shared.camera_stream import ThreadedCameraStream, select_camera_source, normalize_ip_url
from shared.voice_listener import VoiceCommandListener

class MasterAssistiveHub:
    def __init__(self, source=None, ask_camera=False):
        self.tts = TextToSpeechEngine(rate=1, volume=100)
        self.source = source
        self.ask_camera = ask_camera
        self.active_mode = "MENU"  # 'MENU', 'WALKING', 'CURRENCY', 'TEXT'
        self.is_running = True
        self.last_voice_phrase = ""
        self.switch_requested = None
        self.stop_feature_event = threading.Event()

        # Determine camera source
        if self.source is None:
            self.source, self.source_type = select_camera_source(ask=self.ask_camera)
        else:
            if isinstance(self.source, str) and not self.source.isdigit():
                self.source = normalize_ip_url(self.source)
                self.source_type = "IP Camera"
            else:
                self.source = int(self.source)
                self.source_type = "Webcam"

        # Initialize background voice listener
        self.listener = VoiceCommandListener(callback=self._handle_voice_command)

        # Initial speech greeting
        self.tts.speak("AI Assistive Vision online. Say a command like Walking Assistance, Currency, or Read Text.", force=True)

    def _handle_voice_command(self, intent: str, raw_text: str):
        """Callback invoked when a voice command is recognized from the microphone."""
        self.last_voice_phrase = raw_text
        print(f"\n[VOICE COMMAND RECEIVED]: '{raw_text}' -> Intent: {intent.upper()}")

        if intent == "walking":
            if self.active_mode != "WALKING":
                self.tts.speak("Switching to Real-Time Walking Assistance.", force=True)
                self.switch_requested = "WALKING"
                self.stop_feature_event.set()

        elif intent == "vehicle":
            if self.active_mode != "VEHICLE":
                self.tts.speak("Switching to Approaching Vehicle and Traffic Detection.", force=True)
                self.switch_requested = "VEHICLE"
                self.stop_feature_event.set()

        elif intent == "hazard":
            if self.active_mode != "HAZARD":
                self.tts.speak("Switching to Obstacle and Path Hazard Detection.", force=True)
                self.switch_requested = "HAZARD"
                self.stop_feature_event.set()

        elif intent == "person":
            if self.active_mode != "PERSON":
                self.tts.speak("Switching to Familiar Person Recognition.", force=True)
                self.switch_requested = "PERSON"
                self.stop_feature_event.set()

        elif intent == "currency":
            if self.active_mode != "CURRENCY":
                self.tts.speak("Switching to Indian Currency Recognition.", force=True)
                self.switch_requested = "CURRENCY"
                self.stop_feature_event.set()

        elif intent == "text":
            if self.active_mode != "TEXT":
                self.tts.speak("Switching to Text Reading and OCR mode.", force=True)
                self.switch_requested = "TEXT"
                self.stop_feature_event.set()

        elif intent == "help":
            self.tts.speak("You can say: Walking Assistance, Approaching Vehicle, Path Hazard, Person Recognition, Currency Recognition, or Read Text.", force=True)

        elif intent == "menu":
            self.tts.speak("Returned to main menu. Say a command.", force=True)
            self.switch_requested = "MENU"
            self.stop_feature_event.set()

        elif intent == "exit":
            self.tts.speak("Assistive Vision shutting down. Goodbye.", force=True)
            self.is_running = False
            self.switch_requested = "EXIT"
            self.stop_feature_event.set()

    def run(self):
        print("=" * 65)
        print("  AI ASSISTIVE VISION SYSTEM - MASTER VOICE COMMAND HUB")
        print("=" * 65)
        print(f"  Camera Source: {self.source_type} ({self.source})")
        print("\n  Voice Commands (Speak naturally into microphone):")
        print("    🗣️  'Walking Assistance'  ->  Real-Time Walking Assistance (Feature 1)")
        print("    🗣️  'Vehicle' / 'Traffic'  ->  Approaching Vehicle Detection (Feature 2)")
        print("    🗣️  'Hazard' / 'Pothole'   ->  Obstacle & Path Hazard Detection (Feature 3)")
        print("    🗣️  'Person' / 'Friend'   ->  Familiar Person Recognition (Feature 4)")
        print("    🗣️  'Currency'            ->  Indian Currency Recognition (Feature 5)")
        print("    🗣️  'Read Text'           ->  Text Reading & OCR (Feature 6)")
        print("    🗣️  'Help'                ->  List spoken commands")
        print("    🗣️  'Exit' / 'Quit'       ->  Close application")
        print("\n  Manual Hotkey Shortcuts:")
        print("    [1] Walking | [2] Vehicle | [3] Hazard | [4] Person | [5] Currency | [6] Text | [Q] Quit")
        print("=" * 65)

        while self.is_running:
            if self.switch_requested == "WALKING":
                self.switch_requested = None
                self._launch_walking_module()

            elif self.switch_requested == "VEHICLE":
                self.switch_requested = None
                self._launch_vehicle_module()

            elif self.switch_requested == "HAZARD":
                self.switch_requested = None
                self._launch_hazard_module()

            elif self.switch_requested == "PERSON":
                self.switch_requested = None
                self._launch_person_module()

            elif self.switch_requested == "CURRENCY":
                self.switch_requested = None
                self._launch_currency_module()

            elif self.switch_requested == "TEXT":
                self.switch_requested = None
                self._launch_text_reading_module()

            elif self.switch_requested == "EXIT":
                break

            else:
                self._run_menu_dashboard()

        self._shutdown()

    def _run_menu_dashboard(self):
        """Displays the Master Command Hub dashboard window while in menu mode."""
        window_name = "AI Assistive Vision - Master Voice Control Hub"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)

        cap = ThreadedCameraStream(self.source)
        time.sleep(0.3)

        prev_time = time.time()
        fps = 0.0

        try:
            while self.is_running and self.switch_requested is None:
                current_time = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(0.001, (current_time - prev_time)))
                prev_time = current_time

                ret, frame = cap.read()
                if not ret or frame is None:
                    frame = np.full((480, 640, 3), (25, 25, 30), dtype=np.uint8)
                else:
                    frame = cv2.resize(frame, (640, 480))

                # Render Master Hub HUD
                self._draw_master_hud(frame, fps)

                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    self.is_running = False
                    break
                elif key == ord('1'):
                    self._handle_voice_command("walking", "keyboard 1")
                elif key == ord('2'):
                    self._handle_voice_command("vehicle", "keyboard 2")
                elif key == ord('3'):
                    self._handle_voice_command("hazard", "keyboard 3")
                elif key == ord('4'):
                    self._handle_voice_command("person", "keyboard 4")
                elif key == ord('5'):
                    self._handle_voice_command("currency", "keyboard 5")
                elif key == ord('6'):
                    self._handle_voice_command("text", "keyboard 6")
                elif key in [ord('h'), ord('H')]:
                    self._handle_voice_command("help", "keyboard help")

        finally:
            cap.release()
            cv2.destroyWindow(window_name)

    def _draw_master_hud(self, frame: np.ndarray, fps: float):
        h, w = frame.shape[:2]

        # Top Header
        vision_utils.draw_header_hud(frame, title="AI ASSISTIVE VISION - MASTER HUB", subtitle=f"Voice Control Active | {self.source_type}", fps=fps, is_muted=False)

        # Center Voice Command Options Card
        card_w, card_h = w - 60, 320
        card_x, card_y = 30, 65
        vision_utils.draw_transparent_rect(frame, card_x, card_y, card_w, card_h, (15, 20, 25), alpha=0.88, border_color=(0, 220, 255), border_thickness=2)

        cv2.putText(frame, "SAY A VOICE COMMAND OR PRESS NUMBER KEYS:", (card_x + 20, card_y + 24), cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 255, 200), 1, cv2.LINE_AA)
        cv2.line(frame, (card_x + 20, card_y + 30), (card_x + card_w - 20, card_y + 30), (80, 90, 100), 1)

        # Feature rows
        rows = [
            ("1", "Walking Assistance", "'Walking' / 'Navigation' / 'Guide me'", (0, 255, 120)),
            ("2", "Vehicle Detection", "'Vehicle' / 'Traffic' / 'Car'", (0, 180, 255)),
            ("3", "Path Hazard & Stairs", "'Hazard' / 'Pothole' / 'Stairs' / 'Steps'", (255, 140, 0)),
            ("4", "Person Recognition", "'Person' / 'Friend' / 'Who is there'", (255, 80, 180)),
            ("5", "Currency Recognition", "'Currency' / 'Money' / 'Rupees'", (255, 200, 0)),
            ("6", "Text Reading (OCR)", "'Read text' / 'Document' / 'Signboard'", (200, 100, 255)),
            ("H", "Voice Help", "'Help' / 'Commands'", (220, 220, 220))
        ]

        for idx, (num, name, phrase, col) in enumerate(rows):
            ry = card_y + 54 + (idx * 35)
            cv2.rectangle(frame, (card_x + 20, ry - 16), (card_x + 46, ry + 6), col, -1)
            cv2.putText(frame, num, (card_x + 27, ry), cv2.FONT_HERSHEY_DUPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, name, (card_x + 56, ry), cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Say: {phrase}", (card_x + 275, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 200, 220), 1, cv2.LINE_AA)

        # Bottom Microphone Status Banner
        banner_h = 75
        banner_y = h - banner_h - 10
        banner_w = w - 40
        banner_x = 20
        vision_utils.draw_transparent_rect(frame, banner_x, banner_y, banner_w, banner_h, (25, 30, 40), alpha=0.9, border_color=(255, 255, 255), border_thickness=1)

        mic_status = "LISTENING FOR VOICE COMMANDS..."
        cv2.putText(frame, mic_status, (banner_x + 20, banner_y + 30), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 150), 1, cv2.LINE_AA)
        
        last_phrase = self.last_voice_phrase if self.last_voice_phrase else "Ready"
        cv2.putText(frame, f"Last Heard: \"{last_phrase}\"", (banner_x + 20, banner_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 230, 255), 1, cv2.LINE_AA)

    def _launch_walking_module(self):
        """Launches Feature 1: Real-time Walking Assistance."""
        self.active_mode = "WALKING"
        self.stop_feature_event.clear()
        try:
            print("\n[Master Hub] Launching Feature 1: Real-time Walking Assistance...")
            walking_mod = importlib.import_module("01_real_time_walking_assistance.main")
            walking_mod.run_walking_assistant(source=self.source, ask=False, stop_event=self.stop_feature_event)
        except Exception as e:
            print(f"[Master Hub Error] Walking assistance error: {e}")
        finally:
            self.active_mode = "MENU"
            if self.switch_requested is None and self.is_running:
                self.tts.speak("Returned to main menu. Say a command.", force=True)

    def _launch_vehicle_module(self):
        """Launches Feature 2: Approaching Vehicle Detection."""
        self.active_mode = "VEHICLE"
        self.stop_feature_event.clear()
        try:
            print("\n[Master Hub] Launching Feature 2: Approaching Vehicle & Collision Warning...")
            vehicle_mod = importlib.import_module("02_approaching_vehicle_detection.main")
            vehicle_mod.run_vehicle_system(source=self.source, ask=False, stop_event=self.stop_feature_event)
        except Exception as e:
            print(f"[Master Hub Error] Vehicle detection error: {e}")
        finally:
            self.active_mode = "MENU"
            if self.switch_requested is None and self.is_running:
                self.tts.speak("Returned to main menu. Say a command.", force=True)

    def _launch_hazard_module(self):
        """Launches Feature 3: Obstacle & Path Hazard Detection."""
        self.active_mode = "HAZARD"
        self.stop_feature_event.clear()
        try:
            print("\n[Master Hub] Launching Feature 3: Obstacle & Path Hazard Detection...")
            hazard_mod = importlib.import_module("03_obstacle_and_path_hazard_detection.main")
            hazard_mod.run_hazard_system(source=self.source, ask=False, stop_event=self.stop_feature_event)
        except Exception as e:
            print(f"[Master Hub Error] Path hazard detection error: {e}")
        finally:
            self.active_mode = "MENU"
            if self.switch_requested is None and self.is_running:
                self.tts.speak("Returned to main menu. Say a command.", force=True)

    def _launch_person_module(self):
        """Launches Feature 4: Familiar Person Recognition."""
        self.active_mode = "PERSON"
        self.stop_feature_event.clear()
        try:
            print("\n[Master Hub] Launching Feature 4: Familiar Person Recognition...")
            person_mod = importlib.import_module("04_familiar_person_recognition.main")
            person_mod.run_face_recognition_system(source=self.source, ask=False, stop_event=self.stop_feature_event)
        except Exception as e:
            print(f"[Master Hub Error] Person recognition error: {e}")
        finally:
            self.active_mode = "MENU"
            if self.switch_requested is None and self.is_running:
                self.tts.speak("Returned to main menu. Say a command.", force=True)

    def _launch_currency_module(self):
        """Launches Feature 5: Indian Currency Recognition (Uses Laptop Camera)."""
        self.active_mode = "CURRENCY"
        self.stop_feature_event.clear()
        try:
            print("\n[Master Hub] Launching Feature 5: Indian Currency Recognition...")
            currency_mod = importlib.import_module("05_currency_recognition.main")
            currency_mod.run_currency_system(source=None, ask=False, stop_event=self.stop_feature_event)
        except Exception as e:
            print(f"[Master Hub Error] Currency recognition error: {e}")
        finally:
            self.active_mode = "MENU"
            if self.switch_requested is None and self.is_running:
                self.tts.speak("Returned to main menu. Say a command.", force=True)

    def _launch_text_reading_module(self):
        """Launches Feature 6: Text Reading."""
        self.active_mode = "TEXT"
        self.stop_feature_event.clear()
        try:
            print("\n[Master Hub] Launching Feature 6: Text Reading (OCR)...")
            text_mod = importlib.import_module("06_text_reading.main")
            text_mod.run_text_reading_system(source=self.source, ask=False, stop_event=self.stop_feature_event)
        except Exception as e:
            print(f"[Master Hub Error] Text reading error: {e}")
        finally:
            self.active_mode = "MENU"
            if self.switch_requested is None and self.is_running:
                self.tts.speak("Returned to main menu. Say a command.", force=True)

    def _shutdown(self):
        print("\n[Master Hub] Shutting down...")
        self.listener.stop()
        self.tts.shutdown()
        cv2.destroyAllWindows()
        print("[Master Hub] Shutdown complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Voice-Controlled Assistive Vision System")
    parser.add_argument("--source", type=str, default=None, help="Camera index or IP camera stream URL")
    parser.add_argument("--ip", type=str, default=None, help="Phone IP camera stream URL")
    parser.add_argument("--ask", action="store_true", help="Prompt for camera source on startup")
    args = parser.parse_args()

    input_source = args.ip if args.ip is not None else args.source
    if input_source is not None and str(input_source).isdigit():
        input_source = int(input_source)

    hub = MasterAssistiveHub(source=input_source, ask_camera=args.ask)
    hub.run()
