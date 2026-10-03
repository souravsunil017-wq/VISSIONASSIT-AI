"""
Interactive Voice Narration Manager for Feature 6 (Text Reading).
Handles continuous live signboard announcements and interactive document narration
with Play/Pause, Repeat, and Next sentence controls.
"""

import sys
import os
import time
import threading
from typing import List

# Add parent directory to path to allow importing shared modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.audio_engine import TextToSpeechEngine
try:
    from . import config
    from .text_extractor import TextBlock
except ImportError:
    import config
    from text_extractor import TextBlock

class TextReadingVoiceAssistant:
    def __init__(self):
        self.tts = TextToSpeechEngine(rate=config.VOICE_RATE, volume=config.VOICE_VOLUME)
        
        # Live Signboard Spotting State
        self.last_announced_live_text = ""
        self.last_announced_live_time = 0.0

        # Document Reading Mode State
        self.is_document_mode = False
        self.is_paused = False
        self.document_sentences: List[str] = []
        self.current_idx = 0
        self._doc_thread = None
        self._stop_reading_event = threading.Event()

    def announce_live_spot(self, blocks: List[TextBlock]):
        """
        Mode 1: Spot & announce prominent signboards and headings during live camera feed.
        """
        if self.is_document_mode or not blocks:
            return

        now = time.time()
        # Find the most prominent text block (largest area / highest confidence)
        prominent_block = max(blocks, key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1))
        text = prominent_block.text.strip()

        # Ignore if very short or identical to recently announced
        words = text.split()
        if len(words) == 0 or len(words) > config.MAX_LIVE_ANNOUNCE_WORDS:
            return

        if text.lower() == self.last_announced_live_text.lower() and (now - self.last_announced_live_time) < 4.0:
            return

        phrase = f"Text detected: {text}"
        print(f"[LIVE SIGNBOARD]: \"{text}\"")
        self.tts.speak(phrase, force=False, min_interval=config.LIVE_SPOTTER_INTERVAL_SEC)
        
        self.last_announced_live_text = text
        self.last_announced_live_time = now

    def start_document_reading(self, blocks: List[TextBlock]):
        """
        Mode 2: Initiates sequential sentence-by-sentence reading of a captured page.
        """
        self.stop_document_reading()
        if not blocks:
            self.tts.speak("No text found in captured image. Please align camera and try again.", force=True)
            return

        # Combine text blocks into continuous text
        full_text = " ".join([b.text for b in blocks if b.text != "[Text Region Detected]"])
        if not full_text.strip():
            self.tts.speak("Text detected but characters are unclear. Try holding closer.", force=True)
            return

        # Split into sentences (by period, exclamation, question mark, or newlines)
        import re
        sentences = [s.strip() for s in re.split(r'[.!?\n]+', full_text) if len(s.strip()) > 1]
        if not sentences:
            sentences = [full_text]

        self.document_sentences = sentences
        self.current_idx = 0
        self.is_document_mode = True
        self.is_paused = False
        self._stop_reading_event.clear()

        # Start document narrator thread
        self._doc_thread = threading.Thread(target=self._document_reading_worker, daemon=True)
        self._doc_thread.start()

    def _document_reading_worker(self):
        """Worker thread that narrates sentences sequentially."""
        intro = f"Reading document. Found {len(self.document_sentences)} sentences."
        print(f"\n[DOCUMENT READER]: {intro}")
        self.tts.speak(intro, force=True)
        time.sleep(1.8)

        while not self._stop_reading_event.is_set() and self.current_idx < len(self.document_sentences):
            if self.is_paused:
                time.sleep(0.2)
                continue

            sentence = self.document_sentences[self.current_idx]
            print(f"[NARRATING ({self.current_idx + 1}/{len(self.document_sentences)})]: \"{sentence}\"")
            self.tts.speak(sentence, force=True)

            # Wait for utterance to finish speaking
            while self.tts.is_busy() and not self._stop_reading_event.is_set():
                time.sleep(0.1)

            time.sleep(config.SENTENCE_PAUSE_SEC)
            self.current_idx += 1

        if not self._stop_reading_event.is_set() and self.current_idx >= len(self.document_sentences):
            print("[DOCUMENT READER]: End of document reached.")
            self.tts.speak("End of document.", force=True)
            self.is_document_mode = False

    def pause_or_resume(self):
        """Toggle pause/resume during document narration."""
        if not self.is_document_mode:
            return
        self.is_paused = not self.is_paused
        status = "Paused" if self.is_paused else "Resuming"
        print(f"[READER]: {status}")
        self.tts.speak(status, force=True)

    def repeat_sentence(self):
        """Re-reads the previous/current sentence."""
        if not self.is_document_mode or not self.document_sentences:
            return
        self.current_idx = max(0, self.current_idx - 1)
        print(f"[READER]: Repeating sentence {self.current_idx + 1}")

    def next_sentence(self):
        """Skips to the next sentence."""
        if not self.is_document_mode or not self.document_sentences:
            return
        if self.current_idx < len(self.document_sentences) - 1:
            self.current_idx += 1
            print(f"[READER]: Skipped to sentence {self.current_idx + 1}")

    def stop_document_reading(self):
        """Stops document reading mode and returns to live camera mode."""
        self._stop_reading_event.set()
        self.is_document_mode = False
        self.is_paused = False
        self.document_sentences = []
        self.current_idx = 0

    def toggle_mute(self) -> bool:
        return self.tts.toggle_mute()

    @property
    def is_muted(self) -> bool:
        return self.tts.is_muted

    def shutdown(self):
        self.stop_document_reading()
        self.tts.shutdown()
