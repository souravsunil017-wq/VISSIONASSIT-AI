"""
Camera Stream Manager.
Connects to PC Webcam or Mobile Phone IP Camera with zero latency.
"""

import os
import json
import time
import threading
import cv2

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "camera_settings.json"))

def load_camera_config() -> dict:
    default_config = {
        "ip_camera_url": "http://192.168.1.100:8080/video",
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

def select_camera_source(cli_source=None, ask=False):
    """
    Selects camera source from CLI, interactive prompt, or camera_settings.json.
    """
    if cli_source is not None:
        if str(cli_source).isdigit():
            return int(cli_source), "Webcam"
        return normalize_ip_url(str(cli_source)), "IP Camera"

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

    # Default to IP camera from config if set, else webcam
    if cfg.get("ip_camera_url"):
        return cfg.get("ip_camera_url"), "IP Camera"
    return cfg.get("webcam_index", 0), "Webcam"


class ThreadedCameraStream:
    """Threaded camera capture to prevent buffer lag for IP camera streams."""
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.ret = False
        self.frame = None
        self.stopped = False

        if self.cap.isOpened():
            self.ret, self.frame = self.cap.read()
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
        else:
            self.stopped = True

    def _update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                self.stopped = True
                break
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            self.ret = ret
            self.frame = frame
            time.sleep(0.005)

    def read(self):
        return self.ret, self.frame

    def isOpened(self):
        return not self.stopped and self.cap.isOpened()

    def is_opened(self):
        return self.isOpened()

    def release(self):
        self.stopped = True
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        if self.cap.isOpened():
            self.cap.release()

    def stop(self):
        self.release()
