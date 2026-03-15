# backend/cv_engine/buffer.py
from collections import deque
from backend.core.schemas import FrameData

class SlidingWindowBuffer:
    def __init__(self, window_size_sec: int = 5, fps: int = 5):
        self.max_frames = window_size_sec * fps
        # Limit dolunca eski veriyi otomatik silen kuyruk
        self.buffer = deque(maxlen=self.max_frames)

    def add_frame(self, data: FrameData):
        self.buffer.append(data)

    def get_window_data(self) -> list[FrameData]:
        return list(self.buffer)

    def is_ready(self) -> bool:
        """Tamponun (buffer) dolup dolmadığını kontrol eder."""
        return len(self.buffer) == self.max_frames