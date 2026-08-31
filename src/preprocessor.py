"""
Image Preprocessing Utilities
Handles image validation, color space conversion, and resizing for RT-DETR inference.
"""

from __future__ import annotations
from typing import Tuple
import cv2
import numpy as np


class Preprocessor:
    """Handles image preprocessing before model inference."""

    def __init__(
        self,
        image_size: Tuple[int, int] = (640, 640),
        convert_rgb: bool = True,
    ) -> None:
        self.image_size = image_size
        self.convert_rgb = convert_rgb

    @staticmethod
    def validate(frame: np.ndarray) -> None:
        """Validate input OpenCV numpy image frame."""
        if frame is None:
            raise ValueError("Input frame is None.")
        if not isinstance(frame, np.ndarray):
            raise TypeError("Input frame must be a numpy.ndarray.")
        if frame.size == 0:
            raise ValueError("Input frame is empty.")
        if frame.ndim not in (2, 3):
            raise ValueError("Input frame must have 2 or 3 dimensions.")

    def resize(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target dimensions."""
        return cv2.resize(frame, self.image_size, interpolation=cv2.INTER_LINEAR)

    def convert_color(self, frame: np.ndarray) -> np.ndarray:
        """Convert BGR image to RGB if requested."""
        if not self.convert_rgb:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Execute full preprocessing pipeline."""
        self.validate(frame)
        frame = self.resize(frame)
        frame = self.convert_color(frame)
        return frame

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return self.preprocess(frame)