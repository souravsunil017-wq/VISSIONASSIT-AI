"""
Voice Command Listener for Hands-Free Feature Switching.
Listens to microphone in a background thread and classifies spoken intents.
"""

import threading
import time
import re
import json
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Callable, Optional

class _RemoteCommandHandler(BaseHTTPRequestHandler):
    listener_ref = None

    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors()
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._send_cors()
        self.end_headers()
        from shared.audio_engine import TextToSpeechEngine
        resp = {
            "status": "ok",
            "service": "Assist AI PC Controller",
            "pc_audio_muted": TextToSpeechEngine.is_global_muted(),
            "pc_mic_muted": VoiceCommandListener.is_global_mic_muted(),
        }
        self.wfile.write(json.dumps(resp).encode("utf-8"))

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
            data = json.loads(body)
            raw = str(data.get("mode") or data.get("command") or "").strip()
            if raw and self.listener_ref:
                intent = self.listener_ref._classify_intent(raw) or raw.lower()
                print(f"\n[PHONE COMMAND RECEIVED via POST]: '{raw}' -> Intent: {intent.upper()}")
                VoiceCommandListener.handle_control_intent(intent)
                self.listener_ref.last_heard_phrase = f"Phone: {raw}"
                self.listener_ref.last_detected_intent = intent
                if self.listener_ref.callback:
                    self.listener_ref.callback(intent, f"phone:{raw}")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            from shared.audio_engine import TextToSpeechEngine
            resp = {
                "status": "ok",
                "pc_audio_muted": TextToSpeechEngine.is_global_muted(),
                "pc_mic_muted": VoiceCommandListener.is_global_mic_muted(),
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        except Exception:
            self.send_response(500)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress clutter


class VoiceCommandListener:
    """
    Background non-blocking speech recognition & mobile remote command listener.
    Dispatches parsed intent callbacks: 'walking', 'currency', 'text', 'help', 'menu', 'exit'.
    """
    _global_mic_muted: bool = False

    @classmethod
    def set_global_mic_muted(cls, muted: bool):
        cls._global_mic_muted = muted
        print(f"[VoiceListener] PC Microphone Muted: {muted}")

    @classmethod
    def toggle_global_mic_muted(cls) -> bool:
        cls._global_mic_muted = not cls._global_mic_muted
        print(f"[VoiceListener] PC Microphone Muted: {cls._global_mic_muted}")
        return cls._global_mic_muted

    @classmethod
    def is_global_mic_muted(cls) -> bool:
        return cls._global_mic_muted

    @classmethod
    def handle_control_intent(cls, intent: str) -> bool:
        """Executes PC audio and mic mute commands immediately."""
        try:
            from shared.audio_engine import TextToSpeechEngine
        except ImportError:
            return False

        if intent in ["mute_pc_audio", "mute_audio", "silence_pc"]:
            TextToSpeechEngine.set_global_muted(True)
            print("[VoiceListener] PC Speaker Audio is now MUTED.")
            return True
        elif intent in ["unmute_pc_audio", "unmute_audio"]:
            TextToSpeechEngine.set_global_muted(False)
            print("[VoiceListener] PC Speaker Audio is now UNMUTED.")
            return True
        elif intent in ["toggle_pc_audio", "toggle_audio"]:
            is_muted = TextToSpeechEngine.toggle_global_muted()
            print(f"[VoiceListener] PC Speaker Audio toggled to {'MUTED' if is_muted else 'UNMUTED'}.")
            return True
        elif intent in ["mute_pc_mic", "mute_mic"]:
            cls.set_global_mic_muted(True)
            print("[VoiceListener] PC Microphone is now MUTED.")
            return True
        elif intent in ["unmute_pc_mic", "unmute_mic"]:
            cls.set_global_mic_muted(False)
            print("[VoiceListener] PC Microphone is now UNMUTED.")
            return True
        elif intent in ["toggle_pc_mic", "toggle_mic"]:
            is_muted = cls.toggle_global_mic_muted()
            print(f"[VoiceListener] PC Microphone toggled to {'MUTED' if is_muted else 'UNMUTED'}.")
            return True
        elif intent in ["unmute_all", "unmute_both", "cancel_mute", "cancel_muting"]:
            TextToSpeechEngine.set_global_muted(False)
            cls.set_global_mic_muted(False)
            print("[VoiceListener] PC Speaker Audio and Microphone are now BOTH UNMUTED.")
            return True
        elif intent in ["mute_all", "mute_both"]:
            TextToSpeechEngine.set_global_muted(True)
            cls.set_global_mic_muted(True)
            print("[VoiceListener] PC Speaker Audio and Microphone are now BOTH MUTED.")
            return True
        return False

    def __init__(self, callback: Optional[Callable[[str, str], None]] = None):
        self.callback = callback
        self._stop_event = threading.Event()
        self.is_listening = False
        self.last_heard_phrase = ""
        self.last_detected_intent = ""
        self._thread = None
        self._poll_thread = None
        self._http_thread = None
        self.has_microphone = False

        # Start listener threads
        self.start()

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._listen_worker, daemon=True)
        self._thread.start()

        # Start phone remote polling & HTTP server threads
        self._poll_thread = threading.Thread(target=self._poll_phone_worker, daemon=True)
        self._poll_thread.start()

        self._http_thread = threading.Thread(target=self._run_http_listener, daemon=True)
        self._http_thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=0.5)

    def _listen_worker(self):
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 280  # Sensitive to speech
            recognizer.dynamic_energy_threshold = True
            recognizer.pause_threshold = 0.6   # Short pause for quick command response

            # Check if microphone is accessible
            mic = sr.Microphone()
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.8)
            self.has_microphone = True
            print("[Voice Listener] Microphone initialized. Listening for commands...")

            while not self._stop_event.is_set():
                if VoiceCommandListener._global_mic_muted:
                    self.is_listening = False
                    time.sleep(0.25)
                    continue
                try:
                    self.is_listening = True
                    with mic as source:
                        audio = recognizer.listen(source, timeout=1.5, phrase_time_limit=3.5)
                    
                    self.is_listening = False
                    if self._stop_event.is_set():
                        break

                    # Recognize speech using Google Speech Recognition
                    try:
                        spoken_text = recognizer.recognize_google(audio).lower().strip()
                        print(f"\n[MIC HEARD]: \"{spoken_text}\"")
                        self.last_heard_phrase = spoken_text
                        intent = self._classify_intent(spoken_text)

                        if intent:
                            self.last_detected_intent = intent
                            if self.callback:
                                self.callback(intent, spoken_text)
                    except sr.UnknownValueError:
                        pass
                    except sr.RequestError as e:
                        # Network error on Google Speech API fallback
                        pass

                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    time.sleep(0.5)

        except Exception as e:
            self.has_microphone = False
            print(f"[Voice Listener Info] Microphone direct streaming fallback: {e}")
            print("[Voice Listener Info] Keyboard hotkey shortcuts active (1=Walking, 5=Currency, 6=Text).")

    def _classify_intent(self, text: str) -> Optional[str]:
        """Classifies spoken text into actionable feature intents."""
        t = text.lower()

        # 1. Feature 1: Walking Assistance
        if any(w in t for w in ["walk", "walking", "assistance", "navigate", "navigation", "guide me"]):
            return "walking"

        # 2. Feature 2: Approaching Vehicle Detection
        elif any(w in t for w in ["vehicle", "car", "traffic", "approaching", "collision", "bus", "truck"]):
            return "vehicle"

        # 3. Feature 3: Obstacle & Path Hazard Detection (Potholes, Stairs, Overhead)
        elif any(w in t for w in ["hazard", "pothole", "potholes", "stair", "stairs", "step", "steps", "overhead", "branch", "drop", "hole"]):
            return "hazard"

        # 4. Feature 4: Familiar Person Recognition
        elif any(w in t for w in ["person", "face", "family", "friend", "who is"]):
            return "person"

        # 5. Feature 5: Currency Recognition
        elif any(w in t for w in ["currency", "money", "rupee", "rupees", "cash", "note", "notes", "five hundred", "hundred"]):
            return "currency"

        # 6. Feature 6: Text Reading
        elif any(w in t for w in ["read", "text", "ocr", "signboard", "document", "page", "letter", "words"]):
            return "text"

        # 7. Navigation / Control Intents
        elif any(w in t for w in ["help", "commands", "options", "what can i say"]):
            return "help"
        elif any(w in t for w in ["menu", "back", "home", "main menu"]):
            return "menu"
        elif any(w in t for w in ["exit", "quit", "stop", "close", "shut down", "bye"]):
            return "exit"

        # 8. PC Audio & Microphone Muting Controls
        elif any(w in t for w in ["mute pc audio", "mute laptop audio", "mute audio", "silence pc", "mute speaker"]):
            return "mute_pc_audio"
        elif any(w in t for w in ["unmute pc audio", "unmute laptop audio", "unmute audio", "unmute speaker"]):
            return "unmute_pc_audio"
        elif any(w in t for w in ["mute pc mic", "mute laptop mic", "mute mic", "disable pc mic", "disable mic"]):
            return "mute_pc_mic"
        elif any(w in t for w in ["unmute pc mic", "unmute laptop mic", "unmute mic", "enable pc mic", "enable mic"]):
            return "unmute_pc_mic"
        elif any(w in t for w in ["unmute both", "unmute all", "cancel mute", "cancel muting", "turn on both", "enable both", "reset audio"]):
            return "unmute_all"
        elif any(w in t for w in ["mute both", "mute all", "silence both"]):
            return "mute_all"
        elif t in ["mute_pc_audio", "unmute_pc_audio", "toggle_pc_audio", "mute_pc_mic", "unmute_pc_mic", "toggle_pc_mic", "unmute_all", "mute_all", "unmute_both", "mute_both", "cancel_mute"]:
            return t

        return None

    def _poll_phone_worker(self):
        """Continuously pulls voice commands from the phone's port 8080 endpoint."""
        candidate_bases = [
            "http://10.36.171.167:8080",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "http://10.231.110.219:8080",
        ]
        active_base = None

        while not self._stop_event.is_set():
            # Check camera settings for saved phone URL
            try:
                from shared.camera_stream import load_camera_config
                cfg = load_camera_config()
                saved_url = cfg.get("ip_camera_url", "")
                if saved_url:
                    base = saved_url.split("/video")[0].rstrip("/")
                    if base not in candidate_bases:
                        candidate_bases.insert(0, base)
            except Exception:
                pass

            # If no active base, probe candidates
            if not active_base:
                for base in candidate_bases:
                    try:
                        req = urllib.request.Request(f"{base}/status", headers={"User-Agent": "AssistAI-Voice"})
                        with urllib.request.urlopen(req, timeout=0.5) as resp:
                            if resp.status == 200:
                                active_base = base
                                print(f"[Voice Listener] Connected to Phone Command Channel at: {active_base}")
                                break
                    except Exception:
                        pass
                if not active_base:
                    time.sleep(1.0)
                    continue

            # Poll for commands
            try:
                req = urllib.request.Request(f"{active_base}/poll_command", headers={"User-Agent": "AssistAI-Voice"})
                with urllib.request.urlopen(req, timeout=0.8) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        if data.get("has_command") and data.get("command"):
                            raw = str(data["command"]).strip()
                            intent = self._classify_intent(raw) or raw.lower()
                            print(f"\n[PHONE COMMAND RECEIVED]: '{raw}' -> Intent: {intent.upper()}")
                            VoiceCommandListener.handle_control_intent(intent)
                            self.last_heard_phrase = f"Phone: {raw}"
                            self.last_detected_intent = intent
                            if self.callback:
                                self.callback(intent, f"phone:{raw}")
            except urllib.error.URLError:
                # Phone might have momentarily disconnected
                active_base = None
                time.sleep(0.5)
            except Exception:
                time.sleep(0.2)

            time.sleep(0.12)

    def _run_http_listener(self):
        """Starts a background HTTP listener on port 5000 for push commands."""
        try:
            handler_class = _RemoteCommandHandler
            handler_class.listener_ref = self
            server = HTTPServer(("0.0.0.0", 5000), handler_class)
            while not self._stop_event.is_set():
                server.handle_request()
        except Exception:
            # If port 5000 is occupied or blocked, the poll worker still handles everything
            pass
