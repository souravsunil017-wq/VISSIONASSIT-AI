"""
Hazard Voice Guidance & Priority Alert System.
Provides immediate, prioritized, non-blocking voice warnings for potholes, stairs, overhead obstacles, and slippery floors.
"""

import sys
import os
import time
from typing import List, Optional

# Add parent directory to path to allow importing shared modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.audio_engine import TextToSpeechEngine

try:
    from . import config
    from .pothole_detector import PotholeHazard
    from .staircase_detector import StaircaseHazard
    from .overhead_detector import OverheadHazard, WetSurfaceHazard
except ImportError:
    import config
    from pothole_detector import PotholeHazard
    from staircase_detector import StaircaseHazard
    from overhead_detector import OverheadHazard, WetSurfaceHazard

class HazardVoiceAssistant:
    def __init__(self):
        self.tts = TextToSpeechEngine(rate=config.VOICE_RATE, volume=config.VOICE_VOLUME)
        self.last_announced_pothole_time = 0.0
        self.last_announced_stair_time = 0.0
        self.last_announced_overhead_time = 0.0
        self.last_announced_wet_time = 0.0
        self.last_global_alert_time = 0.0

    def evaluate_and_announce_hazards(self,
                                      potholes: List[PotholeHazard],
                                      stair: Optional[StaircaseHazard],
                                      overhead: Optional[OverheadHazard],
                                      wet_surface: Optional[WetSurfaceHazard]):
        """
        Evaluates detected path hazards with prioritized audio announcements.
        Priority Hierarchy:
        1. Low Overhead Obstacles (immediate head-strike risk)
        2. Immediate Stairs Down (fall risk)
        3. Potholes / Cavities (trip / ankle-twist risk)
        4. Stairs Up (navigation assistance)
        5. Wet / Slick Surfaces (slip caution)
        """
        now = time.time()

        # 1. Priority: Overhead Hazards (Immediate Head Strike Danger)
        if overhead is not None and (now - self.last_announced_overhead_time) >= config.ALERT_COOLDOWN_OVERHEAD:
            if (now - self.last_global_alert_time) >= 1.5:
                phrase = "Warning! Low-hanging obstacle overhead. Please duck."
                print(f"[VOICE - OVERHEAD ALERT]: \"{phrase}\"")
                self.tts.speak(phrase, force=True)
                self.last_announced_overhead_time = now
                self.last_global_alert_time = now
                return

        # 2. Priority: Stairs Down (Fall / Drop Hazard)
        if stair is not None and stair.stair_type == "Stairs Down":
            if (now - self.last_announced_stair_time) >= config.ALERT_COOLDOWN_STAIRS:
                if (now - self.last_global_alert_time) >= 1.5:
                    phrase = f"Caution. Stairs down ahead, {stair.distance_meters} meters."
                    print(f"[VOICE - STAIRS DOWN ALERT]: \"{phrase}\"")
                    self.tts.speak(phrase, force=True)
                    self.last_announced_stair_time = now
                    self.last_global_alert_time = now
                    return

        # 3. Priority: Potholes / Ground Drop-offs
        if potholes:
            closest_pothole = potholes[0]
            if (now - self.last_announced_pothole_time) >= config.ALERT_COOLDOWN_POTHOLE:
                if (now - self.last_global_alert_time) >= 1.8:
                    dir_phrase = "ahead" if closest_pothole.corridor == "Center" else f"on your {closest_pothole.corridor.lower()}"
                    phrase = f"Warning. Pothole or drop-off {dir_phrase}, {closest_pothole.distance_meters} meters."
                    print(f"[VOICE - POTHOLE ALERT]: \"{phrase}\"")
                    self.tts.speak(phrase, force=False, min_interval=config.ALERT_COOLDOWN_POTHOLE)
                    self.last_announced_pothole_time = now
                    self.last_global_alert_time = now
                    return

        # 4. Priority: Stairs Up
        if stair is not None and stair.stair_type == "Stairs Up":
            if (now - self.last_announced_stair_time) >= config.ALERT_COOLDOWN_STAIRS:
                if (now - self.last_global_alert_time) >= 2.0:
                    phrase = f"Stairs going up ahead, {stair.distance_meters} meters."
                    print(f"[VOICE - STAIRS UP ALERT]: \"{phrase}\"")
                    self.tts.speak(phrase, force=False, min_interval=config.ALERT_COOLDOWN_STAIRS)
                    self.last_announced_stair_time = now
                    self.last_global_alert_time = now
                    return

        # 5. Priority: Wet / Slick Surfaces
        if wet_surface is not None and (now - self.last_announced_wet_time) >= config.ALERT_COOLDOWN_WET:
            if (now - self.last_global_alert_time) >= 2.5:
                phrase = "Caution. Wet slippery floor detected ahead."
                print(f"[VOICE - WET FLOOR ALERT]: \"{phrase}\"")
                self.tts.speak(phrase, force=False, min_interval=config.ALERT_COOLDOWN_WET)
                self.last_announced_wet_time = now
                self.last_global_alert_time = now

    def toggle_mute(self) -> bool:
        return self.tts.toggle_mute()

    @property
    def is_muted(self) -> bool:
        return self.tts.is_muted

    def shutdown(self):
        self.tts.shutdown()
