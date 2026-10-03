# Shared Utility Modules

This directory contains centralized, reusable infrastructure components utilized across all 6 features of the AI Assistive Vision System.

---

## 1. Modules Overview

```
d:\Ashik\project\shared/
├── __init__.py
├── audio_engine.py      # Thread-safe Windows SAPI.SpVoice non-blocking speech synthesizer
├── camera_stream.py     # High-performance multi-threaded IP camera & webcam stream grabber
├── object_detector.py   # Universal OpenCV DNN object detector (SSD MobileNet V3)
├── vision_utils.py      # Glassmorphic HUD overlay renderer, zone corridors, and direction banners
├── voice_listener.py    # Background microphone listener for voice intent navigation
└── README.md            # Architecture documentation & Viva Q&A
```

---

## 2. Detailed Technical Breakdown

### 1. `camera_stream.py` (Zero-Latency Frame Acquisition)
- **Problem**: OpenCV's default `cv2.VideoCapture` uses an internal frame buffer. When streaming over Wi-Fi (HTTP/MJPEG/RTSP) from a smartphone, any processing latency in the main thread causes frames to buffer, introducing a 2-5 second visual delay.
- **Solution**: `ThreadedCameraStream` initiates an independent daemon thread that continuously grabs frames from the network buffer, storing only the single most recent frame. Calls to `cap.read()` always return the fresh live frame with **0 ms buffer delay**.

### 2. `audio_engine.py` (Non-Blocking Speech Synthesis)
- **Problem**: Standard text-to-speech engines block the calling thread during utterance duration (1-3 seconds), causing video frame rate to drop to 0 FPS.
- **Solution**: `TextToSpeechEngine` uses **Windows Native SAPI (`SAPI.SpVoice`)** initialized via `pythoncom.CoInitialize()` running on a dedicated worker thread with a bounded task queue and `is_busy()` mutex locking.

### 3. `vision_utils.py` (Augmented Reality HUD & UI Rendering)
- Renders translucent overlay bounding boxes, status pills, FPS counters, Left/Center/Right ground corridor zones, and high-visibility directional navigation arrows.

---

## 3. Shared Modules Viva Voce Q&A

### Q1: Why did you separate these utilities into a `shared/` folder?
> **Answer**:  
> To follow the **DRY (Don't Repeat Yourself)** principle and modular software engineering best practices. All 6 planned features (Walking Assistance, Vehicle Detection, Hazard Detection, Person Recognition, Currency Recognition, Text Reading) need camera streaming, speech synthesis, and HUD rendering. Centralizing them ensures uniform performance, code reusability, and easy maintenance.

### Q2: How does `ThreadedCameraStream` handle sudden Wi-Fi disconnections?
> **Answer**:  
> The background thread catches connection errors and sets `stopped = True`. The `isOpened()` and `read()` methods return `(False, None)`, allowing the main feature loops to safely handle reconnections or prompt the user without crashing.
