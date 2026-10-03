"""
Currency Voice Assistant for INR Denomination Announcements & Wallet Tally.
"""

import sys
import os
import time

# Add parent directory to path to allow importing shared modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.audio_engine import TextToSpeechEngine
try:
    from . import config
    from .classifier_engine import CurrencyResult
except ImportError:
    import config
    from classifier_engine import CurrencyResult

class CurrencyVoiceAssistant:
    def __init__(self):
        self.tts = TextToSpeechEngine(rate=config.VOICE_RATE, volume=config.VOICE_VOLUME)
        self.last_announced_denom = ""
        self.last_announced_time = 0.0
        self.wallet_total = 0
        self.counted_notes = []

    def announce_denomination(self, result: CurrencyResult, add_to_wallet: bool = False):
        """
        Announces the recognized denomination clearly to the visually impaired user.
        """
        now = time.time()
        if not result.is_valid_note:
            return

        denom = result.denomination
        conf_pct = int(result.final_confidence * 100)

        # Check cooldown to prevent repetitive spamming of the same banknote
        if denom == self.last_announced_denom and (now - self.last_announced_time) < config.ANNOUNCEMENT_COOLDOWN:
            return

        # Formulate clear verbal announcement
        phrase = f"{denom} Rupees detected."
        
        if add_to_wallet:
            try:
                val = int(denom)
                self.wallet_total += val
                self.counted_notes.append(val)
                phrase += f" Added {denom} Rupees. Total wallet balance: {self.wallet_total} Rupees."
            except ValueError:
                pass

        print(f"[VOICE]: \"{phrase}\" (Confidence: {conf_pct}%)")
        self.tts.speak(phrase, force=False, min_interval=config.ANNOUNCEMENT_COOLDOWN)

        self.last_announced_denom = denom
        self.last_announced_time = now

    def speak_total(self):
        """Speaks the current wallet total aloud."""
        phrase = f"Total wallet balance is {self.wallet_total} Rupees across {len(self.counted_notes)} notes."
        print(f"[VOICE]: \"{phrase}\"")
        self.tts.speak(phrase, force=True)

    def reset_wallet(self):
        """Resets the wallet tally."""
        self.wallet_total = 0
        self.counted_notes = []
        phrase = "Wallet tally has been reset to zero Rupees."
        print(f"[VOICE]: \"{phrase}\"")
        self.tts.speak(phrase, force=True)

    def toggle_mute(self) -> bool:
        return self.tts.toggle_mute()

    @property
    def is_muted(self) -> bool:
        return self.tts.is_muted

    def shutdown(self):
        self.tts.shutdown()
