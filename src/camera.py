"""
Unified Camera & Video Source Manager
Supports webcams, video files, RTSP/HTTP streams, and static image inputs.
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np


class Camera:
    """
    Unified video & image source manager for the RT-DETR detection pipeline.
    """

    def __init__(
        self,
        source: Union[int, str, Path] = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
        buffer_size: int = 1,
        reconnect: bool = False,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 2.0,
    ) -> None:
        self.source = source
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = fps
        self.buffer_size = max(1, int(buffer_size))
        self.reconnect = reconnect
        self.reconnect_attempts = max(0, int(reconnect_attempts))
        self.reconnect_delay = max(0.0, float(reconnect_delay))

        self.capture: Optional[cv2.VideoCapture] = None
        self._opened = False
        self._frame_count = 0
        self._last_frame_time = 0.0
        self._current_fps = 0.0
        self._is_image_file = self._check_if_image()
        self._image_read_done = False

    def _check_if_image(self) -> bool:
        """Check if source points to a static image file."""
        if isinstance(self.source, (str, Path)):
            src_str = str(self.source).lower()
            return src_str.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")) and Path(self.source).is_file()
        return False

    def _prepare_source(self) -> Union[int, str]:
        if isinstance(self.source, str) and self.source.strip().isdigit():
            return int(self.source.strip())
        if isinstance(self.source, Path):
            return str(self.source)
        return self.source

    def open(self) -> bool:
        """Open the configured camera, video, stream, or image source."""
        if self._opened:
            return True

        if self._is_image_file:
            self._opened = True
            self._image_read_done = False
            return True

        source = self._prepare_source()
        self.capture = cv2.VideoCapture(source)

        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            self._opened = False
            return False

        if self.requested_width:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.requested_width))
        if self.requested_height:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.requested_height))
        if self.requested_fps:
            self.capture.set(cv2.CAP_PROP_FPS, float(self.requested_fps))
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        self._opened = True
        self._frame_count = 0
        self._last_frame_time = time.perf_counter()
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from source.

        Returns
        -------
        (success, frame) : (bool, np.ndarray | None)
        """
        if not self._opened:
            if not self.open():
                return False, None

        if self._is_image_file:
            if not self._image_read_done:
                frame = cv2.imread(str(self.source))
                if frame is not None:
                    self._image_read_done = True
                    self._frame_count = 1
                    return True, frame
            return False, None

        if self.capture is None:
            return False, None

        success, frame = self.capture.read()
        if success and frame is not None:
            self._frame_count += 1
            self._update_fps()
            return True, frame

        # Handle stream reconnection if applicable
        if self.reconnect and not isinstance(self.source, (int, Path)):
            for _ in range(self.reconnect_attempts):
                self.close()
                time.sleep(self.reconnect_delay)
                if self.open():
                    return self.read()

        return False, None

    def _update_fps(self) -> None:
        now = time.perf_counter()
        if self._last_frame_time > 0:
            elapsed = now - self._last_frame_time
            if elapsed > 0:
                inst_fps = 1.0 / elapsed
                self._current_fps = inst_fps if self._current_fps == 0 else 0.9 * self._current_fps + 0.1 * inst_fps
        self._last_frame_time = now

    @property
    def is_open(self) -> bool:
        if self._is_image_file:
            return self._opened and not self._image_read_done
        return self._opened and self.capture is not None and self.capture.isOpened()

    @property
    def width(self) -> int:
        if self._is_image_file and Path(self.source).exists():
            img = cv2.imread(str(self.source))
            return img.shape[1] if img is not None else 0
        return int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.capture else 0

    @property
    def height(self) -> int:
        if self._is_image_file and Path(self.source).exists():
            img = cv2.imread(str(self.source))
            return img.shape[0] if img is not None else 0
        return int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.capture else 0

    @property
    def fps(self) -> float:
        if self._current_fps > 0:
            return self._current_fps
        if self.capture:
            src_fps = self.capture.get(cv2.CAP_PROP_FPS)
            return float(src_fps) if src_fps > 0 else 30.0
        return 30.0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def close(self) -> None:
        """Close capture device and release resources."""
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass
            self.capture = None
        self._opened = False

    def information(self) -> Dict[str, Any]:
        """Return camera telemetry and configuration."""
        return {
            "source": str(self.source),
            "is_open": self.is_open,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "frames_read": self.frame_count,
        }

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()