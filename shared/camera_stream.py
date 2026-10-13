"""
Camera Stream Manager.
Connects to PC Webcam or Mobile Phone IP Camera with zero latency.
Automatically detects Assist AI phone camera stream across USB, Wi-Fi, and ADB.
"""

import os
import sys
import json
import time
import threading
import subprocess
import urllib.request
import cv2
import numpy as np

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "camera_settings.json"))

def load_camera_config() -> dict:
    default_config = {
        "ip_camera_url": "http://10.36.171.167:8080/video",
        "webcam_index": 0
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                default_config.update(data)
        except Exception:
            pass
    return default_config

def save_camera_config(config_dict: dict):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
    except Exception:
        pass

def normalize_ip_url(url_input: str) -> str:
    url = url_input.strip()
    if not url.startswith("http://") and not url.startswith("https://") and not url.startswith("rtsp://"):
        url = f"http://{url}/video"
    return url

def auto_detect_camera_url() -> str:
    """
    Auto-detects the Assist AI phone camera IP stream.
    Tests candidate URLs (USB NCM link, ADB forward, hotspot, network gateways).
    Returns working URL or None.
    """
    candidates = [
        "http://10.36.171.167:8080/video",  # Direct USB NCM link (fastest)
        "http://localhost:8080/video",       # Localhost via ADB port forward
        "http://10.231.110.219:8080/video",  # Phone Hotspot IP
    ]

    # Dynamically discover any active network gateway (e.g. phone over USB cable)
    try:
        output = subprocess.check_output("route print 0.0.0.0", shell=True, text=True, stderr=subprocess.DEVNULL)
        for line in output.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0] == "0.0.0.0":
                gw = parts[2]
                if gw != "0.0.0.0":
                    gw_url = f"http://{gw}:8080/video"
                    if gw_url not in candidates:
                        candidates.insert(0, gw_url)
    except Exception:
        pass

    # Include saved URL from configuration
    cfg = load_camera_config()
    saved_url = cfg.get("ip_camera_url")
    if saved_url and saved_url not in candidates:
        candidates.append(saved_url)

    # Probe each candidate endpoint with quick timeout
    for url in candidates:
        status_url = url.replace("/video", "/status")
        try:
            req = urllib.request.Request(status_url, headers={"User-Agent": "AIAssistiveVision"})
            with urllib.request.urlopen(req, timeout=0.6) as resp:
                if resp.status == 200:
                    print(f"[CameraStream] Auto-detected Assist AI phone camera at: {url}")
                    cfg["ip_camera_url"] = url
                    save_camera_config(cfg)
                    return url
        except Exception:
            pass

        # Fallback probe directly on /video
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AIAssistiveVision"})
            with urllib.request.urlopen(req, timeout=0.6) as resp:
                if resp.status == 200:
                    print(f"[CameraStream] Auto-detected Assist AI phone camera at: {url}")
                    cfg["ip_camera_url"] = url
                    save_camera_config(cfg)
                    return url
        except Exception:
            pass

    return None

def select_camera_source(cli_source=None, ask=False):
    """
    Selects camera source from CLI, auto-detected phone stream, or interactive prompt.
    """
    if cli_source is not None:
        if str(cli_source).isdigit():
            return int(cli_source), "Webcam"
        return normalize_ip_url(str(cli_source)), "IP Camera"

    # 1. Attempt automatic discovery of Assist AI phone stream
    detected_url = auto_detect_camera_url()
    if detected_url:
        return detected_url, "IP Camera"

    # 2. If not detected automatically and ask is True, prompt the user
    cfg = load_camera_config()

    if ask:
        print("\n" + "=" * 50)
        print("          SELECT CAMERA SOURCE")
        print("=" * 50)
        print(f" [1] PC Webcam (Index {cfg.get('webcam_index', 0)})")
        print(f" [2] Saved IP Camera ({cfg.get('ip_camera_url')})")
        print(f" [3] Enter New IP Camera Address")
        print("=" * 50)

        try:
            choice = input("Select [1-3] (Default: 2): ").strip()
        except Exception:
            choice = "2"

        if choice == "1":
            return cfg.get("webcam_index", 0), "Webcam"
        elif choice == "3":
            raw_ip = input("Enter IP Address (e.g. 192.168.1.5:8080 or full URL): ").strip()
            if raw_ip:
                formatted_url = normalize_ip_url(raw_ip)
                cfg["ip_camera_url"] = formatted_url
                save_camera_config(cfg)
                return formatted_url, "IP Camera"
            return cfg.get("ip_camera_url"), "IP Camera"
        else:
            return cfg.get("ip_camera_url"), "IP Camera"

    # 3. Default to saved IP camera from config if set, else webcam
    if cfg.get("ip_camera_url"):
        return cfg.get("ip_camera_url"), "IP Camera"
    return cfg.get("webcam_index", 0), "Webcam"


class ThreadedCameraStream:
    """
    High-performance threaded camera capture with dynamic resolution switching support.
    Supports both standard USB webcams and IP camera streams (MJPEG).
    Seamlessly adapts to on-the-fly resolution and orientation changes (Portrait <-> Landscape)
    without buffer lag or freezing.
    """
    def __init__(self, src=0):
        self.src = src
        self.ret = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()

        is_ip = isinstance(self.src, str) and (
            self.src.startswith("http://") or self.src.startswith("https://") or self.src.startswith("rtsp://")
        )
        self.is_ip = is_ip

        if self.is_ip:
            self.thread = threading.Thread(target=self._ip_worker, daemon=True)
            self.thread.start()
        else:
            cam_idx = int(self.src) if str(self.src).isdigit() else 0
            self.cap = cv2.VideoCapture(cam_idx)
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            self.thread = threading.Thread(target=self._cv_worker, daemon=True)
            self.thread.start()

        # Wait up to 3.0 seconds for network connection and first frame delivery
        for _ in range(30):
            if self.ret and self.frame is not None:
                break
            time.sleep(0.1)

    def _ip_worker(self):
        """Dedicated background thread parsing MJPEG streams directly with dynamic resolution handling."""
        while not self.stopped:
            try:
                req = urllib.request.Request(
                    self.src,
                    headers={"User-Agent": "AIAssistiveVision"}
                )
                stream = urllib.request.urlopen(req, timeout=3.0)
                buf = b""
                while not self.stopped:
                    chunk = stream.read(8192)
                    if not chunk:
                        break
                    buf += chunk
                    while True:
                        # Find JPEG Start-Of-Image
                        soi = buf.find(b"\xff\xd8")
                        if soi == -1:
                            buf = buf[-1:] if buf else b""
                            break
                        if soi > 0:
                            buf = buf[soi:]
                            soi = 0
                        # Find JPEG End-Of-Image
                        eoi = buf.find(b"\xff\xd9", 2)
                        if eoi == -1:
                            break
                        jpg_bytes = buf[:eoi + 2]
                        buf = buf[eoi + 2:]
                        # Decode frame dynamically (handles 480x720 <-> 720x480 resolution switches on-the-fly)
                        frame = cv2.imdecode(
                            np.frombuffer(jpg_bytes, dtype=np.uint8),
                            cv2.IMREAD_COLOR
                        )
                        if frame is not None and frame.size > 0:
                            with self.lock:
                                self.ret = True
                                self.frame = frame
            except Exception:
                time.sleep(0.4)

    def _cv_worker(self):
        """Dedicated worker for local USB webcams."""
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                if self.stopped:
                    break
                time.sleep(0.05)
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame
            time.sleep(0.005)

        try:
            self.cap.release()
        except Exception:
            pass

    def read(self):
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def isOpened(self):
        if self.stopped:
            return False
        if self.is_ip:
            return self.ret
        return hasattr(self, "cap") and self.cap.isOpened()

    def is_opened(self):
        return self.isOpened()

    def release(self):
        self.stopped = True

    def stop(self):
        self.release()

