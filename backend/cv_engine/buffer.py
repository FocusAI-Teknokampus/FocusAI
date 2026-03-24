"""
CV Engine — Frame Buffer
Her frame'den gelen CameraSignal'leri geçici olarak tutar.
Feature Extractor buradan okur, State Model'e iletir.
"""

import time
from collections import deque
from threading import Lock
from typing import Optional, List


class FrameBuffer:
    """
    Thread-safe ring buffer.
    Varsayılan: son 30 saniyenin sinyallerini tutar (15 FPS × 30 = 450 frame).
    """

    def __init__(self, maxlen: int = 450):
        self._buf: deque = deque(maxlen=maxlen)
        self._lock = Lock()

    def push(self, signal: dict) -> None:
        """Yeni bir CameraSignal dict'ini buffer'a ekle."""
        with self._lock:
            self._buf.append({"ts": time.time(), "signal": signal})

    def latest(self) -> Optional[dict]:
        """En son sinyali döndür, buffer boşsa None."""
        with self._lock:
            return self._buf[-1]["signal"] if self._buf else None

    def last_n(self, n: int) -> List[dict]:
        """Son n sinyali liste olarak döndür."""
        with self._lock:
            items = list(self._buf)
        return [item["signal"] for item in items[-n:]]

    def since(self, seconds: float) -> List[dict]:
        """Son X saniye içindeki sinyalleri döndür."""
        cutoff = time.time() - seconds
        with self._lock:
            return [item["signal"] for item in self._buf if item["ts"] >= cutoff]

    def clear(self):
        with self._lock:
            self._buf.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)
