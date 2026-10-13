# AI Assistive Vision System for Visually Impaired

An autonomous multi-feature computer vision and voice guidance system designed to empower visually impaired individuals with real-time navigation, safety hazard alerts, person recognition, currency detection, and text reading.

![Project Roadmap](image.png)

---

## 📁 Complete 6-Feature Architecture & Documentation

Each feature is organized in its own standalone module directory with dedicated implementation code, real-time XAI visual HUDs, asynchronous speech engines, and a complete **Teacher Report & 10 Viva Voce Q&As**:

| Module / Folder | Feature Name | Core Computer Vision / AI Technology | Status | Viva Voce Examination Guide |
|---|---|---|---|---|
| [`01_real_time_walking_assistance/`](01_real_time_walking_assistance/README.md) | **Real-Time Walking Assistance** | Lightweight OpenCV DNN SSD MobileNet V3 detection, monocular ground-plane distance triangulation, 3-corridor spatial obstacle analysis & directional walking cues (*Walk Straight, Steer Left, Steer Right, Stop*). | ✅ Completed | [View Report & 10 Viva Q&As](01_real_time_walking_assistance/README.md#5-comprehensive-viva-voce--oral-exam-questions--answers) |
| [`02_approaching_vehicle_detection/`](02_approaching_vehicle_detection/README.md) | **Approaching Vehicle Detection** | Multi-target optical area expansion tracking $\frac{dA}{dt}$, Time-to-Collision (TTC) physics model, and priority voice crash alerts. | ✅ Completed | [View Report & 10 Viva Q&As](02_approaching_vehicle_detection/README.md#7-comprehensive-viva-voce--oral-examination-questions--answers) |
| [`03_obstacle_and_path_hazard_detection/`](03_obstacle_and_path_hazard_detection/README.md) | **Obstacle & Path-Hazard Detection** | Multi-zone spatial CV: Floor adaptive contrast cavities (potholes), Sobel + Hough periodic steps (stairs up/down), eye-level edge density (low branches), and glint reflection (wet floors). | ✅ Completed | [View Report & 10 Viva Q&As](03_obstacle_and_path_hazard_detection/README.md#7-comprehensive-viva-voce--oral-examination-questions--answers) |
| [`04_familiar_person_recognition/`](04_familiar_person_recognition/README.md) | **Familiar Person Recognition** | Deep metric learning (YuNet face detector + SFace 128-D cosine distance matcher) to recognize family/friends with spatial direction & distance tags. | ✅ Completed | [View Report & 10 Viva Q&As](04_familiar_person_recognition/README.md#5-viva-voce-examination-questions--answers-10-qas) |
| [`05_currency_recognition/`](05_currency_recognition/README.md) | **Indian Currency Recognition** | Pre-trained PyTorch MobileNetV3 CNN trained on Kaggle Indian Currency dataset with 4-frame Temporal Consistency Filter, audio announcement, and cash wallet tally. | ✅ Completed | [View Report & 10 Viva Q&As](05_currency_recognition/README.md#6-comprehensive-viva-voce--oral-examination-questions--answers) |
| [`06_text_reading/`](06_text_reading/README.md) | **Text Reading (OCR & TTS)** | Dual-mode OCR reading: (1) Live signboard/heading spotter, and (2) Interactive document sentence narrator with Pause/Repeat/Next speech controls. | ✅ Completed | [View Report & 10 Viva Q&As](06_text_reading/README.md#6-comprehensive-viva-voce--oral-exam-questions--answers) |
| [`shared/`](shared/README.md) | **Shared Infrastructure** | Reusable zero-latency camera streaming (`camera_stream.py`), dual-channel audio engine with mobile sync (`audio_engine.py`), and HUD utilities (`vision_utils.py`). | ✅ Active | [View Shared Docs](shared/README.md) |

---

## 🎙️ Master Hands-Free Voice Controller & Remote Hub (`main.py`)

You can run the unified master application and switch between all 6 features using your **voice**, hotkeys, or through remote mobile commands:

```bash
# Launch Master Voice Control Hub:
python main.py
```

### Supported Voice & Control Commands:
| Say into Microphone / Remote Command | Action | Hotkey |
|---|---|---|
| *"Walking assistance"* / *"Start walking"* / *"Guide me"* | Switches to **Feature 1: Walking Assistance** | `1` |
| *"Vehicle"* / *"Traffic"* / *"Car approaching"* | Switches to **Feature 2: Approaching Vehicle Detection** | `2` |
| *"Hazard"* / *"Pothole"* / *"Stairs"* / *"Steps"* | Switches to **Feature 3: Obstacle & Path Hazard Detection** | `3` |
| *"Person"* / *"Friend"* / *"Who is there"* / *"Face"* | Switches to **Feature 4: Familiar Person Recognition** | `4` |
| *"Currency"* / *"Check money"* / *"Rupees"* / *"Cash"* | Switches to **Feature 5: Currency Recognition** | `5` |
| *"Read text"* / *"Read document"* / *"Signboard"* | Switches to **Feature 6: Text Reading (OCR)** | `6` |
| *"Mute PC audio"* / *"Silence PC"* | Mutes PC speakers (spoken guidance continues on phone) | — |
| *"Unmute PC audio"* | Reactivates PC laptop speakers | — |
| *"Mute PC mic"* | Pauses PC microphone (avoids dual listening with phone) | — |
| *"Unmute PC mic"* | Reactivates PC microphone listening | — |
| *"Unmute all"* / *"Reset audio"* | Unmutes both PC speakers and PC microphone | — |
| *"Help"* / *"What can I say"* | Speaks all available voice commands aloud | `H` |
| *"Back"* / *"Main menu"* | Returns to Master Voice Hub | `ESC` |
| *"Exit"* / *"Quit"* | Shuts down application cleanly | `Q` |

---

## 📱 Mobile Companion Integration & Dual-Channel Audio

The vision hub features tight integration with the **Assist AI** mobile companion application:

* **Real-Time Spoken Audio Forwarding**: Spoken guidance from all 6 vision modules is automatically dispatched via HTTP `POST /speak` to the connected smartphone. Users hear directional navigation alerts and hazard warnings directly through phone speakers or connected Bluetooth earphones.
* **Independent Muting Controls**: Laptop audio can be muted independently while mobile announcements remain active for discreet public navigation.
* **Bidirectional Remote Control**: An embedded background listener (port 5000) and phone polling channel (port 8080 `/poll_command`) accept mode switches and control commands dispatched directly from the mobile app.

---

## 📷 Dynamic Camera Streaming & Auto-Discovery

The camera subsystem (`shared/camera_stream.py`) supports both local USB webcams and mobile phone IP camera feeds:

* **Automatic Stream Discovery**: Probes candidate network interfaces (USB NCM tethering `10.36.171.167:8080`, local ADB port forward `localhost:8080`, mobile hotspot, and network default gateway) to connect to phone cameras without manual IP configuration. Discovered streams are saved to `camera_settings.json`.
* **Zero-Buffer Dynamic MJPEG Reader**: High-speed frame parsing (`urllib` + `cv2.imdecode`) dynamically adapts to on-the-fly resolution and orientation switching (Portrait $\leftrightarrow$ Landscape) with zero lag, eliminating video stream lockups.
* **Persistent Configuration**: Stream URLs and default webcam indexes are saved in `camera_settings.json` across application restarts.

---

## 🚀 Individual Feature Direct Launchers

All features automatically use your configured IP camera or webcam from `camera_settings.json`:

```bash
# Feature 1: Real-Time Walking Assistance
python -m 01_real_time_walking_assistance.main

# Feature 2: Approaching Vehicle Detection
python -m 02_approaching_vehicle_detection.main

# Feature 3: Obstacle & Path Hazard Detection
python -m 03_obstacle_and_path_hazard_detection.main

# Feature 4: Familiar Person Recognition
python -m 04_familiar_person_recognition.main

# Feature 5: Indian Currency Recognition
python -m 05_currency_recognition.main

# Feature 6: Text Reading (OCR)
python -m 06_text_reading.main
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/souravsunil017-wq/VISSIONASSIT-AI.git
   cd VISSIONASSIT-AI
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Audio & Microphone (Optional)**:
   ```bash
   python test_mic.py
   ```

4. **Launch the Master Hub**:
   ```bash
   python main.py
   ```
