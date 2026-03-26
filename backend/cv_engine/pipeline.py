"""
CV Engine pipeline.

Supports both:
1. direct camera capture via OpenCV
2. processing externally provided frames (for browser camera uploads)
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Optional

import numpy as np

from .buffer import FrameBuffer
from .extractors import EmotionExtractor, GazeExtractor, GestureExtractor

W_DROWSY = 0.35
W_HEAD_DOWN = 0.20
W_YAWN = 0.20
W_GAZE = 0.15
W_EAR_LOW = 0.07
W_HAND = 0.03

MA_WINDOW = 30
RISE_SPEED = 0.02
FALL_SPEED = 0.08
MAR_MOUTH_COMBO = 0.36


class _ScoreState:
    def __init__(self):
        self.buffer: deque[float] = deque(maxlen=MA_WINDOW)
        self.smoothed_score = 1.0
        self.lock = threading.Lock()


class CameraSignal:
    def __init__(
        self,
        active: bool,
        gaze: Optional[dict],
        gesture: Optional[dict],
        emotion: Optional[dict],
        timestamp: float,
        frame_id: int,
        processing_ms: float,
        score_state: Optional[_ScoreState] = None,
    ):
        self.active = active
        self.gaze = gaze
        self.gesture = gesture
        self.emotion = emotion
        self.timestamp = timestamp
        self.frame_id = frame_id
        self.processing_ms = processing_ms
        self._score_state = score_state or _ScoreState()

        self._raw_score: Optional[float] = None
        self._attention_score: Optional[float] = None
        self._state = "UNKNOWN"
        self._compute()

    def _compute(self):
        if not self.active or self.gaze is None or self.gesture is None:
            self._state = "UNKNOWN"
            return

        gaze = self.gaze
        gesture = self.gesture

        is_yawning = gaze.get("is_yawning", False)
        if gesture.get("hand_on_mouth") and gaze.get("mar", 0.0) > MAR_MOUTH_COMBO:
            is_yawning = True

        penalty = 0.0
        if gaze.get("is_drowsy"):
            penalty += W_DROWSY
        if is_yawning:
            penalty += W_YAWN
        if gesture.get("is_head_down"):
            penalty += W_HEAD_DOWN
        if gaze.get("gaze_direction") in ("left", "right", "down"):
            penalty += W_GAZE
        if gaze.get("ear_avg", 0.3) < 0.25 and not gaze.get("is_drowsy"):
            penalty += W_EAR_LOW
        if gesture.get("hand_on_chin"):
            penalty += W_HAND
        if gesture.get("hand_on_mouth") and not is_yawning:
            penalty += W_HAND * 0.5
        penalty += gesture.get("scratch_penalty", 0.0)

        if self.emotion and self.emotion.get("is_distracted_emotion"):
            penalty += 0.10

        raw = round(max(0.0, min(1.0, 1.0 - penalty)), 3)
        self._raw_score = raw

        with self._score_state.lock:
            self._score_state.buffer.append(raw)
            target = float(np.mean(self._score_state.buffer))
            diff = target - self._score_state.smoothed_score
            if diff > 0:
                self._score_state.smoothed_score += min(diff, RISE_SPEED)
            else:
                self._score_state.smoothed_score += max(diff, -FALL_SPEED)
            self._score_state.smoothed_score = round(
                max(0.0, min(1.0, self._score_state.smoothed_score)),
                3,
            )
            smoothed = self._score_state.smoothed_score

        self._attention_score = smoothed

        if gaze.get("is_drowsy") or (is_yawning and smoothed < 0.5):
            self._state = "SLEEPY"
        elif smoothed >= 0.70:
            self._state = "FOCUSED"
        elif smoothed >= 0.40:
            self._state = "DISTRACTED"
        else:
            self._state = "SLEEPY"

    @property
    def attention_score(self) -> Optional[float]:
        return self._attention_score

    @property
    def raw_score(self) -> Optional[float]:
        return self._raw_score

    @property
    def state(self) -> str:
        return self._state

    def to_dict(self) -> dict:
        return {
            "active": self.active,
            "gaze": self.gaze,
            "gesture": self.gesture,
            "emotion": self.emotion,
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "processing_ms": self.processing_ms,
            "raw_score": self._raw_score,
            "attention_score": self._attention_score,
            "state": self._state,
        }


def _fallback(score_state: Optional[_ScoreState] = None) -> CameraSignal:
    return CameraSignal(
        active=False,
        gaze=None,
        gesture=None,
        emotion=None,
        timestamp=time.time(),
        frame_id=0,
        processing_ms=0.0,
        score_state=score_state,
    )


class CVPipeline:
    def __init__(
        self,
        camera_index: int = 0,
        target_fps: int = 15,
        use_emotion: bool = False,
        on_signal: Optional[Callable[[CameraSignal], None]] = None,
    ):
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.use_emotion = use_emotion
        self.on_signal = on_signal

        self._gaze = GazeExtractor()
        self._gesture = GestureExtractor()
        self._emotion = EmotionExtractor() if use_emotion else None

        self._buffer = FrameBuffer()
        self._cap = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_id = 0
        self._error_count = 0
        self._extractors_ready = False
        self._score_state = _ScoreState()

        self._latest: CameraSignal = _fallback(score_state=self._score_state)
        self._lock = threading.Lock()

    def start(self) -> bool:
        if self._running:
            return True

        if not self._initialize_extractors():
            return False

        if not self._open_camera():
            return False

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[CVPipeline] Baslatildi - kamera {self.camera_index} @ {self.target_fps} FPS")
        return True

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        self._thread = None
        self._release_camera()
        self._gaze.release()
        self._gesture.release()
        if self._emotion:
            self._emotion.release()
        self._extractors_ready = False
        print("[CVPipeline] Durduruldu.")

    def latest(self) -> CameraSignal:
        with self._lock:
            return self._latest

    def process_frame(self, frame_bgr: np.ndarray) -> CameraSignal:
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            signal = _fallback(score_state=self._score_state)
            with self._lock:
                self._latest = signal
            return signal

        if not self._initialize_extractors():
            signal = _fallback(score_state=self._score_state)
            with self._lock:
                self._latest = signal
            return signal

        signal = self._build_signal(frame_bgr)

        with self._lock:
            self._latest = signal

        self._buffer.push(signal.to_dict())
        if self.on_signal:
            try:
                self.on_signal(signal)
            except Exception as exc:
                print(f"[CVPipeline] Callback hatasi: {exc}")
        return signal

    def buffer(self) -> FrameBuffer:
        return self._buffer

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._latest.active

    def _initialize_extractors(self) -> bool:
        if self._extractors_ready:
            return True

        ok_gaze = self._gaze.initialize()
        ok_gesture = self._gesture.initialize()
        if not ok_gaze or not ok_gesture:
            print("[CVPipeline] Extractor baslatilamadi. Kamerasiz modda devam.")
            self._extractors_ready = False
            return False

        if self.use_emotion and self._emotion:
            self._emotion.initialize()

        self._extractors_ready = True
        return True

    def _open_camera(self) -> bool:
        try:
            import cv2

            self._cap = cv2.VideoCapture(self.camera_index)
            if not self._cap.isOpened():
                self._cap = None
                print(f"[CVPipeline] Kamera {self.camera_index} acilamadi. Kamerasiz modda devam.")
                return False
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            return True
        except Exception as exc:
            print(f"[CVPipeline] Kamera hatasi: {exc}")
            return False

    def _release_camera(self):
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _loop(self):
        interval = 1.0 / self.target_fps

        while self._running:
            started = time.time()

            if not self._cap or not self._cap.isOpened():
                self._error_count += 1
                if self._error_count > 10:
                    print("[CVPipeline] Kamera baglantisi kesildi.")
                    break
                time.sleep(0.5)
                continue

            ret, frame_bgr = self._cap.read()
            if not ret or frame_bgr is None:
                self._error_count += 1
                continue
            self._error_count = 0

            signal = self._build_signal(frame_bgr)

            with self._lock:
                self._latest = signal

            self._buffer.push(signal.to_dict())
            if self.on_signal:
                try:
                    self.on_signal(signal)
                except Exception as exc:
                    print(f"[CVPipeline] Callback hatasi: {exc}")

            sleep = interval - (time.time() - started)
            if sleep > 0:
                time.sleep(sleep)

        self._release_camera()

    def _build_signal(self, frame_bgr: np.ndarray) -> CameraSignal:
        import cv2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        started = time.time()
        gaze = self._gaze.extract(frame_rgb)
        gesture = self._gesture.extract(frame_rgb)
        emotion = self._emotion.extract(frame_bgr) if self._emotion else None
        processing_ms = (time.time() - started) * 1000

        self._frame_id += 1
        return CameraSignal(
            active=True,
            gaze=gaze,
            gesture=gesture,
            emotion=emotion,
            timestamp=time.time(),
            frame_id=self._frame_id,
            processing_ms=round(processing_ms, 1),
            score_state=self._score_state,
        )
