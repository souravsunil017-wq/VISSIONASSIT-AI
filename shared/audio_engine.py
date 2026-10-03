"""
Centralized Non-Blocking Audio Speech Engine.
Uses native Windows SAPI.SpVoice with busy-state management and non-blocking worker thread.
"""

import threading
import queue
import time
import sys

class TextToSpeechEngine:
    def __init__(self, rate: int = 1, volume: int = 100):
        # SAPI Rate: -10 to 10 (1 is clear standard speed), Volume: 0 to 100
        self.rate = rate
        self.volume = volume
        self._speech_queue = queue.Queue(maxsize=3)
        self._stop_event = threading.Event()
        self._is_muted = False
        self._is_speaking = False
        self._last_spoken_text = ""
        self._last_spoken_time = 0.0

        self._worker_thread = threading.Thread(target=self._run_worker, daemon=True)
        self._worker_thread.start()

    def _run_worker(self):
        sapi_voice = None
        pyttsx_engine = None

        # 1. Native Windows SAPI (most robust)
        if sys.platform == "win32":
            try:
                import pythoncom
                import win32com.client
                pythoncom.CoInitialize()
                sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
                sapi_voice.Rate = self.rate
                sapi_voice.Volume = self.volume
            except Exception as e:
                print(f"[AudioEngine Warning] Native SAPI init failed: {e}")
                sapi_voice = None

        # 2. Fallback to pyttsx3
        if sapi_voice is None:
            try:
                import pyttsx3
                pyttsx_engine = pyttsx3.init()
                pyttsx_engine.setProperty('rate', 175)
                pyttsx_engine.setProperty('volume', 1.0)
            except Exception as e:
                print(f"[AudioEngine Warning] pyttsx3 init failed: {e}")

        while not self._stop_event.is_set():
            try:
                text, force = self._speech_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._is_muted:
                self._speech_queue.task_done()
                continue

            self._is_speaking = True
            try:
                if sapi_voice is not None:
                    # SVSFDefault = 0 (speaks full sentence cleanly in worker thread)
                    sapi_voice.Speak(text, 0)
                elif pyttsx_engine is not None:
                    pyttsx_engine.say(text)
                    pyttsx_engine.runAndWait()
                else:
                    print(f"[SPEECH]: {text}")
            except Exception as e:
                print(f"[AudioEngine Error] Failed to speak: {e}")
            finally:
                self._is_speaking = False
                self._speech_queue.task_done()

    def is_busy(self) -> bool:
        """Returns True if the engine is currently speaking or has queued speech."""
        return self._is_speaking or not self._speech_queue.empty()

    def speak(self, text: str, force: bool = False, min_interval: float = 2.0):
        """
        Queue text for speaking. Drops repetitive requests while speech is currently active.
        """
        now = time.time()
        text_clean = text.strip()
        if not text_clean:
            return

        # If already speaking and not forced, let the current utterance finish
        if self.is_busy() and not force:
            return

        # Suppress identical sentence repeats within interval
        if not force and text_clean == self._last_spoken_text and (now - self._last_spoken_time) < min_interval:
            return

        self._last_spoken_text = text_clean
        self._last_spoken_time = now

        # If forced, clear old queue
        if force:
            while not self._speech_queue.empty():
                try:
                    self._speech_queue.get_nowait()
                    self._speech_queue.task_done()
                except Exception:
                    break

        try:
            self._speech_queue.put_nowait((text_clean, force))
        except queue.Full:
            pass

    def toggle_mute(self) -> bool:
        self._is_muted = not self._is_muted
        return self._is_muted

    @property
    def is_muted(self) -> bool:
        return self._is_muted

    def shutdown(self):
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.5)
