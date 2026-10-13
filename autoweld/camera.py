"""Continuously refreshed USB camera frames, never cached as fresh evidence."""

import threading
import time
from pathlib import Path

import cv2


class Camera:
    def __init__(self, device="/dev/video0"):
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open USB camera {device}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.frame = None
        self.timestamp = 0.0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 5
        while not self.fresh and time.monotonic() < deadline:
            time.sleep(0.05)
        if not self.fresh:
            self.close()
            raise RuntimeError("USB camera did not deliver a frame")

    def _run(self):
        while not self.stop.is_set():
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame
                    self.timestamp = time.monotonic()
            else:
                time.sleep(0.05)

    @property
    def fresh(self):
        return self.frame is not None and time.monotonic() - self.timestamp < 1.0

    def rgb(self):
        if not self.fresh:
            raise RuntimeError("USB camera frame is stale")
        with self.lock:
            return cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

    def save(self, path):
        rgb = self.rgb()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Cannot write camera evidence to {path}")

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        self.cap.release()
