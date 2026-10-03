"""
Interactive Microphone Diagnostic & Audio Level Tester.
Tests hardware microphone input, records sample, measures volume levels, and tests Speech Recognition.
"""

import sys
import os
import time
import wave
import audioop

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import speech_recognition as sr
from shared.audio_engine import TextToSpeechEngine

def run_mic_test():
    print("=" * 65)
    print("      🎙️ MICROPHONE & VOICE RECOGNITION DIAGNOSTIC TEST")
    print("=" * 65)

    tts = TextToSpeechEngine(rate=1, volume=100)

    # 1. Scan available audio input devices
    print("\n[Step 1/4] Scanning Audio Input Devices...")
    mics = sr.Microphone.list_microphone_names()
    print(f"Total audio endpoints detected: {len(mics)}\n")

    input_devices = []
    for idx, name in enumerate(mics):
        if any(term in name.lower() for term in ["mic", "input", "headset", "array", "realtek", "buds"]):
            input_devices.append((idx, name))
            print(f"  🎙️ Device [{idx}]: {name}")

    # 2. Check default mic
    print("\n" + "-" * 65)
    print("[Step 2/4] Testing Default System Microphone...")
    
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 250
    recognizer.dynamic_energy_threshold = True

    try:
        mic = sr.Microphone()
        print("Calibrating ambient noise (please remain quiet for 1 second)...")
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
        print(f"✓ Calibration complete. Base Energy Threshold: {int(recognizer.energy_threshold)}")
    except Exception as e:
        print(f"❌ Failed to open default microphone: {e}")
        return

    # 3. Audio Level & Recording Test
    print("\n" + "-" * 65)
    print("[Step 3/4] Testing Microphone Sound Level (Speak or tap mic now!)...")
    tts.speak("Microphone test. Please speak now.", force=True)

    print("\n🎙️ [LISTENING FOR 5 SECONDS - SPEAK NOW!]")
    print("👉 Say something like: 'Walking assistance', 'Currency', or 'Testing 1 2 3'")
    print("-" * 65)

    try:
        with mic as source:
            audio = recognizer.listen(source, timeout=6.0, phrase_time_limit=5.0)
        
        # Check raw audio amplitude / loudness
        raw_data = audio.get_raw_data()
        rms = audioop.rms(raw_data, 2)
        print(f"\n📊 Audio Captured! RMS Volume Level: {rms}")
        if rms < 100:
            print("⚠️ WARNING: Audio volume is very low (< 100). Check if your mic is muted or volume is low in Windows Sound Settings.")
        else:
            print(f"✓ Strong audio signal detected (Volume score: {rms})")

        # Save wav for verification
        wav_path = os.path.join(os.path.dirname(__file__), "test_mic_sample.wav")
        with open(wav_path, "wb") as f:
            f.write(audio.get_wav_data())
        print(f"✓ Recorded audio sample saved to: {wav_path}")

        # 4. Speech Recognition Test
        print("\n" + "-" * 65)
        print("[Step 4/4] Sending to Speech Recognition Engine...")
        recognized_text = recognizer.recognize_google(audio)
        
        print("\n" + "=" * 65)
        print(f"  🎉 SUCCESS! Recognized Speech: \"{recognized_text}\"")
        print("=" * 65)

        # Check intent classification
        from shared.voice_listener import VoiceCommandListener
        listener_mock = VoiceCommandListener.__new__(VoiceCommandListener)
        intent = listener_mock._classify_intent(recognized_text)
        if intent:
            print(f"  🎯 Mapped System Intent: [{intent.upper()}] (Feature trigger verified!)")
        else:
            print(f"  ℹ️ Spoken phrase recognized, but did not match a specific feature keyword.")

        tts.speak(f"Microphone working! I heard: {recognized_text}", force=True)
        time.sleep(3.0)

    except sr.WaitTimeoutError:
        print("\n⚠️ [Timeout]: No sound was detected within 6 seconds.")
        print("Suggestions:")
        print(" 1. Check Windows Settings -> Privacy & Security -> Microphone (Allow apps to access your mic).")
        print(" 2. Make sure your microphone is not physically muted (e.g. headset switch).")
        print(" 3. Check Windows Sound Control Panel -> Recording Devices to make sure default mic is active.")
        tts.speak("No speech detected. Please check microphone settings.", force=True)
        time.sleep(2.5)

    except sr.UnknownValueError:
        print("\n⚠️ [Audio Heard But Speech Unclear]: Audio was captured, but words could not be recognized.")
        print("Tip: Speak clearly into the microphone at normal conversational volume.")
        tts.speak("Audio was detected, but words were unclear. Please speak clearly.", force=True)
        time.sleep(2.5)

    except Exception as e:
        print(f"\n❌ [Recognition Error]: {e}")
        tts.speak("An error occurred during recognition.", force=True)
        time.sleep(2.5)

    finally:
        tts.shutdown()
        print("\n" + "=" * 65)
        print("                 DIAGNOSTIC COMPLETE")
        print("=" * 65)

if __name__ == "__main__":
    run_mic_test()

