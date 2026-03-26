from __future__ import annotations

import base64
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

from backend.core.config import settings
from backend.core.schemas import CameraSignal as StateCameraSignal
from backend.core.schemas import CameraStatusResponse


@dataclass
class _CameraSessionState:
    pipeline: Any
    last_updated_at: Optional[datetime] = None
    last_error: Optional[str] = None
    latest_signal: Optional[StateCameraSignal] = None
    backend_state: Optional[str] = None
    attention_score: Optional[float] = None
    processing_ms: Optional[float] = None
    frame_id: int = 0


class CameraRuntimeService:
    def __init__(self):
        self._sessions: dict[str, _CameraSessionState] = {}
        self._lock = threading.Lock()
        self._stale_after = timedelta(seconds=max(3, settings.window_size_sec))

    def process_frame(self, session_id: str, image_base64: str) -> CameraStatusResponse:
        dependency_error = self._preflight_dependencies()
        if dependency_error is not None:
            return CameraStatusResponse(
                session_id=session_id,
                status="error",
                available=False,
                active=False,
                face_detected=False,
                error=dependency_error,
            )

        state = self._get_or_create_session(session_id)

        try:
            frame_bgr = self._decode_frame(image_base64)
            signal = state.pipeline.process_frame(frame_bgr)
            state.last_updated_at = datetime.utcnow()
            state.backend_state = signal.state
            state.attention_score = signal.attention_score
            state.processing_ms = signal.processing_ms
            state.frame_id = signal.frame_id
            state.latest_signal = self._to_state_camera_signal(signal)
            state.last_error = None

            if signal.frame_id == 0 and signal.processing_ms == 0 and state.latest_signal is None:
                state.last_error = (
                    "CV extractors baslatilamadi. "
                    "numpy/cv2/mediapipe kurulumunu ve backend loglarini kontrol et."
                )
                return self._build_status(
                    session_id,
                    state,
                    available=False,
                    status="error",
                )
        except Exception as exc:
            state.last_updated_at = datetime.utcnow()
            state.last_error = str(exc)
            return self._build_status(session_id, state, available=False, status="error")

        return self._build_status(
            session_id,
            state,
            available=state.latest_signal is not None,
            status="active",
        )

    def get_status(self, session_id: str) -> CameraStatusResponse:
        with self._lock:
            state = self._sessions.get(session_id)

        if state is None:
            return CameraStatusResponse(session_id=session_id)

        status = "active" if self._is_active(state) else "idle"
        available = state.latest_signal is not None and not state.last_error
        if state.last_error:
            status = "error"
            available = False

        return self._build_status(session_id, state, available=available, status=status)

    def get_camera_signal(self, session_id: str) -> Optional[StateCameraSignal]:
        with self._lock:
            state = self._sessions.get(session_id)

        if state is None or not self._is_active(state):
            return None
        return state.latest_signal

    def reset_session(self, session_id: str) -> None:
        with self._lock:
            state = self._sessions.pop(session_id, None)

        if state is not None:
            state.pipeline.stop()

    def _get_or_create_session(self, session_id: str) -> _CameraSessionState:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                from backend.cv_engine import CVPipeline

                state = _CameraSessionState(
                    pipeline=CVPipeline(
                        camera_index=settings.camera_id,
                        target_fps=max(1, settings.fps_target),
                        use_emotion=False,
                    )
                )
                self._sessions[session_id] = state
            return state

    def _preflight_dependencies(self) -> Optional[str]:
        required_modules = (
            ("numpy", "numpy"),
            ("cv2", "opencv-python"),
            ("mediapipe", "mediapipe"),
        )
        for module_name, package_name in required_modules:
            try:
                __import__(module_name)
            except Exception as exc:
                return (
                    f"Eksik veya bozuk CV bagimliligi: {package_name}. "
                    f"Import hatasi: {exc}"
                )
        return None

    def _is_active(self, state: _CameraSessionState) -> bool:
        if state.last_updated_at is None:
            return False
        return datetime.utcnow() - state.last_updated_at <= self._stale_after

    def _build_status(
        self,
        session_id: str,
        state: _CameraSessionState,
        *,
        available: bool,
        status: str,
    ) -> CameraStatusResponse:
        return CameraStatusResponse(
            session_id=session_id,
            status=status,
            available=available,
            active=self._is_active(state),
            face_detected=state.latest_signal is not None,
            backend_state=state.backend_state,
            attention_score=state.attention_score,
            processing_ms=state.processing_ms,
            frame_id=state.frame_id,
            signal=state.latest_signal,
            last_updated_at=state.last_updated_at,
            error=state.last_error,
        )

    def _decode_frame(self, image_base64: str):
        if not image_base64:
            raise ValueError("Bos kamera frame verisi gonderildi.")

        payload = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
        raw_bytes = base64.b64decode(payload)
        if not raw_bytes:
            raise ValueError("Kamera frame'i decode edilemedi.")

        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("OpenCV veya numpy backend ortaminda bulunamadi.") from exc

        buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError("Kamera frame'i goruntuye donusturulemedi.")
        return frame_bgr

    def _to_state_camera_signal(self, signal) -> Optional[StateCameraSignal]:
        gaze = signal.gaze or {}
        gesture = signal.gesture or {}

        if not gaze or not gesture:
            return None

        return StateCameraSignal(
            ear_score=float(gaze.get("ear_avg", 0.0)),
            gaze_on_screen=gaze.get("gaze_direction", "center") == "center",
            hand_on_chin=bool(gesture.get("hand_on_chin", False)),
            head_tilt_angle=float(gesture.get("head_pitch", 0.0)),
        )


camera_runtime_service = CameraRuntimeService()
