"""
Threaded video stream and synthetic frame generator for edge pipelines.
"""

from __future__ import annotations

import time
import threading
from typing import Generator, Tuple, Optional, Union
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class VideoStream:
    """
    Non-blocking threaded video stream reader.
    Decouples frame capture from model inference to prevent latency queuing.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        resolution: Tuple[int, int] = (640, 640),
        buffer_size: int = 4,
        target_fps: int = 30,
    ) -> None:
        self.source = source
        self.resolution = resolution
        self.buffer_size = buffer_size
        self.target_fps = target_fps
        self.frame_delay = 1.0 / max(target_fps, 1)

        self._frame: Optional[np.ndarray] = None
        self._stopped = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._cap = None

    def start(self) -> "VideoStream":
        """Start background frame capture thread."""
        if _HAS_CV2:
            self._cap = cv2.VideoCapture(self.source)
            if self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        self._stopped = False
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        return self

    def _update(self) -> None:
        """Internal capture loop running in daemon thread."""
        synthetic_gen = SyntheticStream(resolution=self.resolution)

        while not self._stopped:
            start_t = time.perf_counter()
            frame = None

            if _HAS_CV2 and self._cap and self._cap.isOpened():
                ret, raw_frame = self._cap.read()
                if ret and raw_frame is not None:
                    frame = raw_frame

            if frame is None:
                # Fallback to simulated high-resolution inspection stream
                frame = synthetic_gen.generate_frame()

            with self._lock:
                self._frame = frame

            elapsed = time.perf_counter() - start_t
            sleep_time = self.frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def read(self) -> Optional[np.ndarray]:
        """Fetch the most recently captured frame (non-blocking)."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self) -> None:
        """Stop background worker and release hardware resources."""
        self._stopped = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if _HAS_CV2 and self._cap:
            self._cap.release()

    def __enter__(self) -> "VideoStream":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


class SyntheticStream:
    """
    Generates synthetic frames simulating industrial assembly line components
    and defects (scratches, cracks, misalignments) for automated testing.
    """

    def __init__(self, resolution: Tuple[int, int] = (640, 640)) -> None:
        self.width, self.height = resolution
        self.frame_idx = 0

    def generate_frame(self) -> np.ndarray:
        """Constructs a deterministic synthetic industrial frame with moving objects."""
        self.frame_idx += 1
        t = self.frame_idx * 0.05

        # Background conveyor surface
        frame = np.full((self.height, self.width, 3), 45, dtype=np.uint8)

        # Draw conveyor belt motion lines
        offset = int((self.frame_idx * 4) % 40)
        for y in range(offset, self.height, 40):
            frame[y : y + 2, :] = 60

        # Draw a synthetic PCB / mechanical component moving across
        comp_cx = int((self.width * 0.5) + np.sin(t) * 120)
        comp_cy = int((self.height * 0.5) + np.cos(t * 0.7) * 40)
        w, h = 220, 160

        x1 = max(0, comp_cx - w // 2)
        y1 = max(0, comp_cy - h // 2)
        x2 = min(self.width, comp_cx + w // 2)
        y2 = min(self.height, comp_cy + h // 2)

        # Component body (greenish metallic)
        frame[y1:y2, x1:x2] = [35, 90, 45]

        # Component pins / contacts (gold color)
        for px in range(x1 + 10, x2 - 10, 25):
            frame[y1 : y1 + 12, px : px + 12] = [212, 175, 55]
            frame[y2 - 12 : y2, px : px + 12] = [212, 175, 55]

        # Synthetic defect: simulated crack or scratch periodically
        if (self.frame_idx // 30) % 2 == 1:
            cx1 = min(x2 - 10, x1 + 40)
            cx2 = min(x2 - 10, x1 + 120)
            cy1 = min(y2 - 10, y1 + 50)
            cy2 = min(y2 - 10, y1 + 90)
            # Reddish defect trace
            frame[cy1:cy2, cx1:cx2] = [40, 40, 210]

        return frame

    def frames(self, count: int = 100) -> Generator[np.ndarray, None, None]:
        """Generate a finite sequence of frames."""
        for _ in range(count):
            yield self.generate_frame()
