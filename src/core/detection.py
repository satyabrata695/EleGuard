"""
Detection Data Model
Defines the Detection dataclass representing a single detected object.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass(slots=True)
class Detection:
    """
    Represents a single detected object.

    Attributes
    ----------
    class_id : int
        Numeric class ID from model.
    class_name : str
        Human-readable class name.
    confidence : float
        Detection confidence score (0.0 to 1.0).
    bbox : tuple[int, int, int, int]
        Bounding box in (x1, y1, x2, y2) pixel coordinates.
    metadata : dict
        Optional metadata.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    metadata: Dict[str, Any] = field(default_factory=dict)

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
    def center(self) -> Tuple[int, int]:
        return (self.x1 + self.width // 2, self.y1 + self.height // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> Dict[str, Any]:
        """Convert Detection object to a dictionary."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Detection":
        """Construct Detection object from dictionary."""
        return cls(
            class_id=int(data["class_id"]),
            class_name=str(data["class_name"]),
            confidence=float(data["confidence"]),
            bbox=tuple(int(v) for v in data["bbox"]),  # type: ignore
            metadata=data.get("metadata", {}),
        )

    def __str__(self) -> str:
        return f"{self.class_name} ({self.confidence:.1%}) {self.bbox}"

    def __repr__(self) -> str:
        return f"Detection(class='{self.class_name}', conf={self.confidence:.3f}, bbox={self.bbox})"