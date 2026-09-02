"""
Detection Data Model
Represents a single detected object in a frame.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(slots=True, frozen=True)
class Detection:
    """
    Represents a single bounding box detection with class, confidence, and spatial attributes.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in absolute pixel coordinates

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x1 + self.width // 2, self.y1 + self.height // 2)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize detection to a dictionary."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "width": self.width,
            "height": self.height,
            "area": self.area,
            "center": list(self.center),
        }

    def __str__(self) -> str:
        return f"{self.class_name.capitalize()} ({self.confidence:.1%}) @ {self.bbox}"
