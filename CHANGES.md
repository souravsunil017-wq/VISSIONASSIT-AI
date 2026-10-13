# AI Eye 1 — Modifications & Integration Changes

This document details all changes and additions made within the `ai_eye1-main` repository to integrate the Python Assistive Vision Hub with the **Assist AI** Flutter mobile application (Pixel 8).

> **Important Note on Core Vision Modules**:  
> None of the 6 core vision model algorithms or feature implementations were modified:
> - `01_real_time_walking_assistance/` (Unchanged)
> - `02_approaching_vehicle_detection/` (Unchanged)
> - `03_obstacle_path_hazard_detection/` (Unchanged)
> - `04_familiar_person_recognition/` (Unchanged)
> - `05_indian_currency_recognition/` (Unchanged)
> - `06_text_reading/` (Unchanged)
>
> All changes are strictly limited to camera input selection, real-time audio forwarding to the phone, and remote control command handling.

---

## Summary of Changed Files

| File | Change Type | Purpose |
|---|---|---|
| [`camera_settings.json`](file:///c:/Users/NihalM/Desktop/C_P/ai_eye1-main/camera_settings.json) | **New File** | Persists IP camera stream URL and default webcam index across runs |
| [`shared/camera_stream.py`](file:///c:/Users/NihalM/Desktop/C_P/ai_eye1-main/shared/camera_stream.py) | **Modified** | Auto-detects phone camera stream and provides non-blocking threaded capture |
| [`shared/audio_engine.py`](file:///c:/Users/NihalM/Desktop/C_P/ai_eye1-main/shared/audio_engine.py) | **Modified** | Forwards spoken guidance to phone via HTTP `POST /speak` and adds PC speaker muting |
| [`shared/voice_listener.py`](file:///c:/Users/NihalM/Desktop/C_P/ai_eye1-main/shared/voice_listener.py) | **Modified** | Embeds HTTP remote control server and adds PC microphone muting controls |
| [`main.py`](file:///c:/Users/NihalM/Desktop/C_P/ai_eye1-main/main.py) | **Modified** | Connects `MasterAssistiveHub` to threaded camera stream and remote command dispatcher |

---

## Detailed File-by-File Changes

### 1. `camera_settings.json` (New File)
- **Path**: `ai_eye1-main/camera_settings.json`
- **Content**:
  ```json
  {
    "ip_camera_url": "http://10.36.171.167:8080/video",
    "webcam_index": 0
  }
  ```
- **Description**: Configuration file created so that once the phone camera stream is discovered or set, the PC software remembers the stream address between runs.

---

### 2. `shared/camera_stream.py` (Modified)
- **Path**: `ai_eye1-main/shared/camera_stream.py`
- **Additions & Modifications**:
  1. **`auto_detect_camera_url() -> str`**:
     - Automatically scans candidate stream endpoints without requiring manual user input:
       - Direct USB NCM cable IP: `http://10.36.171.167:8080/video`
       - ADB reverse / port forward: `http://localhost:8080/video`
       - Phone Hotspot: `http://10.231.110.219:8080/video`
       - Auto-detected default gateway IP (via `route print 0.0.0.0`)
     - Probes `/status` with a 0.6s timeout. If a stream is found, it automatically saves it to `camera_settings.json`.
  2. **`ThreadedCameraStream` Class**:
     - Upgraded with a direct high-speed MJPEG reader (`urllib` + `cv2.imdecode`) for IP camera feeds and threaded capture for webcams.
     - **Dynamic Orientation & Resolution Adaptation**: OpenCV's default `cv2.VideoCapture` FFmpeg backend previously froze and locked up when the phone switched between Portrait (480x720) and Landscape (720x480) mode. The direct parser dynamically handles resolution changes on-the-fly with 0 ms buffer latency and automatic reconnection.
     - Ensures all AI vision models always process the latest live camera frame without freezing or stuttering during orientation changes.
  3. **`select_camera_source(cli_source, ask)`**:
     - Allows selecting between PC Webcam (`0`) and the detected Phone IP Camera stream.
  4. **`load_camera_config()` & `save_camera_config()`**:
     - Manages reading and writing `camera_settings.json`.

---

### 3. `shared/audio_engine.py` (Modified)
- **Path**: `ai_eye1-main/shared/audio_engine.py`
- **Additions & Modifications**:
  1. **`_send_to_phone(self, text: str)`**:
     - Whenever the PC vision models issue an audio announcement (e.g. obstacle distance in walking mode or detected currency notes), `_send_to_phone` is triggered in a background daemon thread.
     - Asynchronously sends an HTTP `POST /speak` request with JSON payload `{"text": text}` to the phone's built-in web server (`http://<phone_ip>:8080/speak`).
     - This allows the blind user to hear walking assistance and hazard alerts directly from the mobile phone's speaker or connected Bluetooth headphones.
  2. **PC Speaker Muting (`set_global_muted`, `toggle_global_muted`, `is_global_muted`)**:
     - Added class-level flag `_global_muted`.
     - In `_run_worker`: When `_global_muted` is True, speech audio from the PC's physical speakers is silenced, but `_send_to_phone` **continues to dispatch speech to the phone**.
     - Allows quiet operation where only the user's phone speaks.

---

### 4. `shared/voice_listener.py` (Modified)
- **Path**: `ai_eye1-main/shared/voice_listener.py`
- **Additions & Modifications**:
  1. **Embedded Remote Control Handler (`_RemoteCommandHandler`)**:
     - Lightweight HTTP server handler added inside the listener thread.
     - **`GET /` & `GET /status`**: Returns PC connection status, active mute states (`pc_audio_muted`, `pc_mic_muted`).
     - **`POST /` & `POST /command`**: Receives mode change and mute control commands sent by the Assist AI phone application.
     - Parses intents like `walking`, `currency`, `hazard`, `text`, `menu`, `mute_pc_audio`, `mute_pc_mic`, `unmute_all`.
  2. **PC Microphone Muting (`set_global_mic_muted`, `is_global_mic_muted`)**:
     - Added class-level flag `_global_mic_muted`.
     - In `_listen_worker`: When `_global_mic_muted` is True, microphone recording is paused so the laptop microphone stops listening. This avoids dual listening when the user prefers phone voice commands.
  3. **`handle_control_intent(intent: str)`**:
     - Directly toggles speaker or microphone muting when control intents are detected.

---

### 5. `main.py` (Modified)
- **Path**: `ai_eye1-main/main.py`
- **Additions & Modifications**:
  1. **`MasterAssistiveHub.__init__`**:
     - Updated camera initialization to use `select_camera_source(ask=self.ask_camera)` instead of assuming camera index `0`.
     - Supports both `ThreadedCameraStream` (for phone MJPEG) and standard integer camera indices.
  2. **Remote Phone Integration in `_handle_voice_command`**:
     - Unified voice recognition callbacks so commands coming from the phone's remote HTTP handler seamlessly set `self.switch_requested` and signal `self.stop_feature_event`.
     - Enables smooth remote mode transitions between walking assistance, vehicle detection, hazard detection, person recognition, currency classification, and text reading.
