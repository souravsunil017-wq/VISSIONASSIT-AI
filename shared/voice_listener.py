"""
Voice Command Listener for Hands-Free Feature Switching.
Listens to microphone in a background thread and classifies spoken intents.
"""

import threading
import time
import re
from typing import Callable, Optional

class VoiceCommandListener:
    """
    Background non-blocking speech recognition listener.
    Dispatches parsed intent callbacks: 'walking', 'currency', 'text', 'help', 'menu', 'exit'.
    """
    def __init__(self, callback: Optional[Callable[[str, str], None]] = None):
        self.callback = callback
        self._stop_event = threading.Event()
        self.is_listening = False
        self.last_heard_phrase = ""
        self.last_detected_intent = ""
        self._thread = None
        self.has_microphone = False

        # Start listener thread
        self.start()

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._listen_worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)

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

        return None
