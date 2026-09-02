"""
Unified Camera & Video Stream Acquisition Module
Supports webcams (device index), video files, image files, and RTSP network streams.
"""

from __future__ import annotations
from pathlib import Path
from typing import Generator, Optional, Tuple, Union
import cv2
import numpy as np


class Camera:
    """
    Manages video stream capture from webcams, files, or RTSP endpoints.
    """

    def __init__(
        self,
        source: Union[int, str, Path] = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        reconnect: bool = True,
        max_reconnect_attempts: int = 5,
    ) -> None:
        self.source = self._normalize_source(source)
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = fps
        self.reconnect = reconnect
        self.max_reconnect_attempts = max_reconnect_attempts

        self._cap: Optional[cv2.VideoCapture] = None
        self._is_image = False
        self._cached_image: Optional[np.ndarray] = None
        self._frame_count = 0
        self._reconnect_count = 0

    @staticmethod
    def _normalize_source(source: Union[int, str, Path]) -> Union[int, str]:
        if isinstance(source, int):
            return source
        source_str = str(source).strip()
        if source_str.isdigit():
            return int(source_str)
        return source_str

    @property
    def is_opened(self) -> bool:
        if self._is_image:
            return self._cached_image is not None
        return self._cap is not None and self._cap.isOpened()

    @property
    def is_file(self) -> bool:
        return isinstance(self.source, str) and not self.source.startswith(("rtsp://", "http://", "https://"))

    @property
    def is_webcam(self) -> bool:
        return isinstance(self.source, int)

    def open(self) -> bool:
        """Open the video capture stream or load the single image."""
        if isinstance(self.source, str) and not self.source.startswith(("rtsp://", "http://", "https://")):
            src_path = Path(self.source)
            if src_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                self._is_image = True
                self._cached_image = cv2.imread(str(src_path))
                return self._cached_image is not None

        self._is_image = False
        self._cap = cv2.VideoCapture(self.source)

        if self.requested_width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        if self.requested_height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        if self.requested_fps:
            self._cap.set(cv2.CAP_PROP_FPS, self.requested_fps)

        return self._cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame."""
        if self._is_image:
            if self._cached_image is not None and self._frame_count == 0:
                self._frame_count += 1
                return True, self._cached_image.copy()
            return False, None

        if self._cap is None or not self._cap.isOpened():
            if self.reconnect and self._reconnect_count < self.max_reconnect_attempts:
                self._reconnect_count += 1
                if self.open():
                    return self.read()
            return False, None

        ret, frame = self._cap.read()
        if ret and frame is not None:
            self._frame_count += 1
            self._reconnect_count = 0
            return True, frame

        return False, None

    def frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Generator yielding sequential (frame_id, frame) tuples."""
        if not self.is_opened:
            if not self.open():
                return

        while True:
            ret, frame = self.read()
            if not ret or frame is None:
                break
            yield self._frame_count, frame

    @property
    def resolution(self) -> Tuple[int, int]:
        if self._is_image and self._cached_image is not None:
            return self._cached_image.shape[1], self._cached_image.shape[0]
        if self._cap and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return 0, 0

    @property
    def fps(self) -> float:
        if self._cap and self._cap.isOpened():
            return self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        return 0.0

    def close(self) -> None:
        """Release underlying video capture handle."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._cached_image = None

    def __enter__(self) -> Camera:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
