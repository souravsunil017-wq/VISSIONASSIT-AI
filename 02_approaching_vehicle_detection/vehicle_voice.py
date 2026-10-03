"""
Vehicle Threat Audio Guidance & Priority Voice Announcer.
Generates urgent, non-blocking verbal collision alerts and spatial warnings for visually impaired users.
"""

import sys
import os
import time
from typing import List, Dict

# Add parent directory to path to allow importing shared modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.audio_engine import TextToSpeechEngine

try:
    from . import config
    from .motion_tracker import VehicleTrack
except ImportError:
    import config
    from motion_tracker import VehicleTrack

class VehicleVoiceAssistant:
    def __init__(self):
        self.tts = TextToSpeechEngine(rate=config.VOICE_RATE, volume=config.VOICE_VOLUME)
        self.last_announced_time: Dict[int, float] = {}
        self.last_global_alert_time = 0.0

    def evaluate_and_announce_threats(self, tracks: List[VehicleTrack]):
        """
        Evaluates all active vehicle tracks and sounds priority-based audio warnings.
        """
        now = time.time()
        
        # Sort tracks by threat severity: CRITICAL first, then WARNING, prioritized by lowest TTC / shortest distance
        critical_threats = [t for t in tracks if t.threat_level == "CRITICAL"]
        warning_threats = [t for t in tracks if t.threat_level == "WARNING"]

        # 1. Handle Critical Emergencies
        if critical_threats:
            # Sort by lowest TTC
            critical_threats.sort(key=lambda t: t.ttc_seconds)
            top_threat = critical_threats[0]
            
            last_time = self.last_announced_time.get(top_threat.track_id, 0.0)
            if (now - last_time) >= config.ALERT_COOLDOWN_CRITICAL or (now - self.last_global_alert_time) >= 1.5:
                direction_phrase = "in front of you" if top_threat.corridor == "Center" else f"on your {top_threat.corridor.lower()}"
                ttc_str = f"{top_threat.ttc_seconds} seconds" if top_threat.ttc_seconds < 10.0 else f"{top_threat.current_distance} meters"
                
                phrase = f"DANGER! {top_threat.class_name.capitalize()} approaching fast {direction_phrase}! {ttc_str}!"
                print(f"[VOICE - CRITICAL ALERT]: \"{phrase}\"")
                
                # Force immediate voice interrupt
                self.tts.speak(phrase, force=True)
                self.last_announced_time[top_threat.track_id] = now
                self.last_global_alert_time = now
            return

        # 2. Handle Warning Level Threats
        if warning_threats:
            warning_threats.sort(key=lambda t: t.ttc_seconds)
            top_warning = warning_threats[0]

            last_time = self.last_announced_time.get(top_warning.track_id, 0.0)
            if (now - last_time) >= config.ALERT_COOLDOWN_WARNING and (now - self.last_global_alert_time) >= 2.5:
                direction_phrase = "ahead" if top_warning.corridor == "Center" else f"on your {top_warning.corridor.lower()}"
                ttc_str = f"{top_warning.ttc_seconds} seconds" if top_warning.ttc_seconds < 10.0 else f"{top_warning.current_distance} meters"
                
                phrase = f"Warning. {top_warning.class_name.capitalize()} approaching {direction_phrase}, {ttc_str}."
                print(f"[VOICE - WARNING ALERT]: \"{phrase}\"")
                
                self.tts.speak(phrase, force=False, min_interval=config.ALERT_COOLDOWN_WARNING)
                self.last_announced_time[top_warning.track_id] = now
                self.last_global_alert_time = now

    def toggle_mute(self) -> bool:
        return self.tts.toggle_mute()

    @property
    def is_muted(self) -> bool:
        return self.tts.is_muted

    def shutdown(self):
        self.tts.shutdown()
