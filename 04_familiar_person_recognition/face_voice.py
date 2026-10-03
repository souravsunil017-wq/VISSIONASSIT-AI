"""
Positional Voice Assistant for Familiar Person Recognition.
Provides spatial audio announcements with anti-repetition cooldowns.
"""

import time
from typing import List, Dict, Optional

try:
    from . import config
    from .face_engine import DetectedFace
except ImportError:
    import config
    from face_engine import DetectedFace

from shared.audio_engine import TextToSpeechEngine

class FaceVoiceAssistant:
    def __init__(self, rate: int = 1, volume: int = 100):
        self.tts = TextToSpeechEngine(rate=rate, volume=volume)
        self.is_muted = False
        # Cooldown trackers: { "PersonName": last_speech_timestamp }
        self.last_spoken_time: Dict[str, float] = {}
        self.last_unknown_time = 0.0

    def toggle_mute(self) -> bool:
        """Toggles audio speech on/off."""
        self.is_muted = not self.is_muted
        status = "Muted" if self.is_muted else "Unmuted"
        print(f"[Voice Assistant] Audio {status}.")
        self.tts.speak(f"Voice {status}.", force=True)
        return self.is_muted

    def process_detected_faces(self, faces: List[DetectedFace]):
        """
        Evaluates detected faces and generates spatial voice feedback.
        """
        if self.is_muted or not faces:
            return

        now = time.time()

        # Prioritize known persons first, then unknown persons
        known_faces = [f for f in faces if f.identity != "Unknown"]
        unknown_faces = [f for f in faces if f.identity == "Unknown"]

        if known_faces:
            # Pick the closest or most central known person
            target = min(known_faces, key=lambda f: f.distance_meters)
            name = target.identity
            last_time = self.last_spoken_time.get(name, 0.0)

            if (now - last_time) >= config.SPEECH_COOLDOWN_PER_PERSON_SEC:
                zone_phrase = {
                    "LEFT": "on your left",
                    "RIGHT": "on your right",
                    "CENTER": "in front of you"
                }.get(target.zone, "nearby")

                dist_str = f"{target.distance_meters} meters away" if target.distance_meters >= 1.0 else "very close"
                phrase = f"{name} is {zone_phrase}, {dist_str}."

                self.tts.speak(phrase)
                self.last_spoken_time[name] = now

        elif unknown_faces:
            if (now - self.last_unknown_time) >= config.UNKNOWN_PERSON_COOLDOWN_SEC:
                target = min(unknown_faces, key=lambda f: f.distance_meters)
                zone_phrase = {
                    "LEFT": "on your left",
                    "RIGHT": "on your right",
                    "CENTER": "in front of you"
                }.get(target.zone, "nearby")

                phrase = f"Unknown person detected {zone_phrase}."
                self.tts.speak(phrase)
                self.last_unknown_time = now

    def speak_enrolled_summary(self, enrolled_names: List[str]):
        """Speaks startup summary of recognized family & friends."""
        if self.is_muted:
            return
        if not enrolled_names:
            self.tts.speak("Face recognition online. No known faces enrolled yet. Add photos to known faces folder.", force=True)
        else:
            names_str = ", ".join(enrolled_names)
            self.tts.speak(f"Face recognition online. Enrolled identities: {names_str}.", force=True)

    def shutdown(self):
        """Cleanly releases TTS audio engine."""
        self.tts.shutdown()
