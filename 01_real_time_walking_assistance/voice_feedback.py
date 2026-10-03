"""
Voice Feedback Manager for Real-time Walking Assistance.
Speaks immediate obstacle distances (meters) and clear spatial positions (Left, In Front, Right).
Ignores far background objects to prevent audio fatigue.
"""

import sys
import os
import time
from collections import deque

# Add parent directory to path to allow importing shared modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.audio_engine import TextToSpeechEngine
try:
    from . import config
    from .path_guidance import NavigationGuidance
except ImportError:
    import config
    from path_guidance import NavigationGuidance

class WalkingVoiceAssistant:
    def __init__(self):
        self.tts = TextToSpeechEngine(rate=config.VOICE_RATE, volume=config.VOICE_VOLUME)
        self.last_announced_phrase = ""
        self.last_announced_time = 0.0

    def process_guidance(self, guidance: NavigationGuidance):
        """
        Translates real-time proximity and spatial decisions into clear directional speech with distances.
        """
        now = time.time()

        # If audio engine is currently speaking, let it finish unless urgent STOP
        if self.tts.is_busy() and not guidance.is_critical:
            return

        phrase = ""
        is_critical = guidance.is_critical
        obj = guidance.primary_object
        dist = guidance.primary_distance
        dist_str = f"{dist:.1f} meters" if dist > 0 else ""

        # 1. CRITICAL STOP (Object in immediate collision range <= 1.25m)
        if guidance.direction_code == "STOP":
            if obj:
                phrase = f"Stop! {obj.capitalize()} close in front, {dist_str}."
            else:
                phrase = "Stop! Path blocked."

        # 2. STEER LEFT / STEER RIGHT (Obstacle in center range 1.25m - 2.6m)
        elif guidance.direction_code in ["LEFT", "RIGHT"]:
            steer_dir = "steer left" if guidance.direction_code == "LEFT" else "steer right"
            if obj and dist > 0:
                phrase = f"{obj.capitalize()} ahead, {dist_str}. Please {steer_dir}."
            elif obj:
                phrase = f"{obj.capitalize()} ahead. Please {steer_dir}."
            else:
                phrase = f"Obstacle ahead. Please {steer_dir}."

        # 3. FORWARD / WALK STRAIGHT (Path ahead is clear of proximate obstacles)
        else:
            if guidance.primary_object and guidance.primary_position and guidance.primary_distance > 0:
                phrase = f"Walk straight. {guidance.primary_object.capitalize()} {guidance.primary_position}, {dist_str}."
            elif guidance.detected_objects_summary:
                phrase = f"Walk straight. {guidance.detected_objects_summary.capitalize()}."
            else:
                phrase = "Path clear. Walk straight."

        # Adaptive cooldown intervals
        interval = config.CRITICAL_ALERT_COOLDOWN if is_critical else (
            config.ANNOUNCEMENT_COOLDOWN if phrase != self.last_announced_phrase else 3.8
        )

        if phrase and (now - self.last_announced_time) >= interval:
            print(f"[VOICE]: \"{phrase}\"")
            self.tts.speak(phrase, force=is_critical, min_interval=interval)
            self.last_announced_phrase = phrase
            self.last_announced_time = now

    def toggle_mute(self) -> bool:
        return self.tts.toggle_mute()

    @property
    def is_muted(self) -> bool:
        return self.tts.is_muted

    def shutdown(self):
        self.tts.shutdown()
