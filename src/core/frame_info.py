"""
FrameInfo Data Model
Stores all telemetry and detection information related to a processed frame.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import numpy as np

from src.core.detection import Detection


@dataclass(slots=True)
class FrameInfo:
    """
    Represents a single processed frame with detections and performance metrics.
    """

    frame_id: int
    frame: np.ndarray
    detections: List[Detection] = field(default_factory=list)
    fps: float = 0.0
    inference_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def detection_count(self) -> int:
        """Return the number of detections in this frame."""
        return len(self.detections)

    @property
    def has_detection(self) -> bool:
        """Return True if at least one object was detected."""
        return len(self.detections) > 0

    @property
    def average_confidence(self) -> float:
        """Calculate average confidence of all detections in this frame."""
        if not self.detections:
            return 0.0
        return sum(det.confidence for det in self.detections) / len(self.detections)

    @property
    def largest_detection(self) -> Optional[Detection]:
        """Return the detection with the largest bounding box area."""
        if not self.detections:
            return None
        return max(self.detections, key=lambda det: det.area)

    def add_detection(self, detection: Detection) -> None:
        """Add a detection to this frame."""
        self.detections.append(detection)

    def clear(self) -> None:
        """Clear all detections in this frame."""
        self.detections.clear()

    def __len__(self) -> int:
        return len(self.detections)

    def __str__(self) -> str:
        return f"Frame {self.frame_id} | {self.detection_count} detections | {self.fps:.1f} FPS | {self.inference_time:.1f}ms"