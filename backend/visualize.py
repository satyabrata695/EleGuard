"""
Detection Visualization Module
Renders bounding boxes, confidence badges, HUD telemetry, and alert overlays on frames.
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np

from backend.core.detection import Detection
from backend.core.frame_info import FrameInfo


class Visualizer:
    """
    Renders detection results and HUD telemetry onto frames.
    """

    CLASS_COLORS = [
        (40, 200, 40),   # Vibrant Green (Elephant)
        (255, 140, 0),   # Amber/Orange
        (30, 144, 255),  # Dodger Blue
        (255, 69, 0),    # Red-Orange
        (186, 85, 211),  # Medium Orchid
        (0, 215, 255),   # Gold/Yellow
    ]

    def __init__(
        self,
        box_thickness: int = 2,
        font_scale: float = 0.55,
        font_thickness: int = 1,
        show_confidence: bool = True,
        show_fps: bool = True,
        show_inference_time: bool = True,
        show_detection_count: bool = True,
        show_class_name: bool = True,
        show_alert_banner: bool = True,
    ) -> None:
        self.box_thickness = max(1, int(box_thickness))
        self.font_scale = max(0.3, float(font_scale))
        self.font_thickness = max(1, int(font_thickness))
        self.show_confidence = show_confidence
        self.show_fps = show_fps
        self.show_inference_time = show_inference_time
        self.show_detection_count = show_detection_count
        self.show_class_name = show_class_name
        self.show_alert_banner = show_alert_banner
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def _get_color(self, detection: Detection) -> Tuple[int, int, int]:
        """Deterministic color per class ID."""
        return self.CLASS_COLORS[detection.class_id % len(self.CLASS_COLORS)]

    def draw_detection(self, frame: np.ndarray, detection: Detection) -> None:
        """Draw a single detection bounding box and label badge."""
        x1, y1, x2, y2 = detection.bbox
        color = self._get_color(detection)

        # Draw main bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness, cv2.LINE_AA)

        # Build label text
        parts = []
        if self.show_class_name:
            parts.append(detection.class_name.capitalize())
        if self.show_confidence:
            parts.append(f"{detection.confidence:.1%}")
        label = " | ".join(parts) if parts else ""

        if label:
            (w, h), baseline = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
            badge_y1 = max(0, y1 - h - baseline - 6)
            badge_y2 = y1
            badge_x2 = min(frame.shape[1], x1 + w + 10)

            # Filled label background badge
            cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
            # Text inside badge
            cv2.putText(
                frame,
                label,
                (x1 + 5, badge_y2 - baseline - 2),
                self.font,
                self.font_scale,
                (0, 0, 0),
                self.font_thickness,
                cv2.LINE_AA,
            )

    def draw_hud(self, frame: np.ndarray, frame_info: FrameInfo) -> None:
        """Draw real-time HUD telemetry on the top-left corner."""
        hud_items = []
        if self.show_fps and frame_info.fps > 0:
            hud_items.append(f"FPS: {frame_info.fps:.1f}")
        if self.show_inference_time and frame_info.inference_time > 0:
            hud_items.append(f"Latency: {frame_info.inference_time:.1f}ms")
        if self.show_detection_count:
            hud_items.append(f"Elephants: {frame_info.detection_count}")

        if not hud_items:
            return

        text = "  |  ".join(hud_items)
        (tw, th), baseline = cv2.getTextSize(text, self.font, 0.5, 1)

        # Translucent dark HUD background pill
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (20 + tw, 20 + th + baseline), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # HUD text
        cv2.putText(frame, text, (15, 15 + th), self.font, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

    def draw_alert_banner(self, frame: np.ndarray, count: int) -> None:
        """Draw a high-visibility warning banner when elephants are detected."""
        banner_text = f"WARNING: {count} ELEPHANT{'S' if count > 1 else ''} DETECTED"
        h, w = frame.shape[:2]
        (tw, th), baseline = cv2.getTextSize(banner_text, self.font, 0.7, 2)

        # Alert banner overlay on top right
        bx1 = w - tw - 30
        by1 = 10
        bx2 = w - 10
        by2 = 20 + th + baseline

        overlay = frame.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        cv2.putText(
            frame,
            banner_text,
            (bx1 + 10, by1 + th + 4),
            self.font,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def render(self, frame_info: FrameInfo) -> np.ndarray:
        """Render all annotations onto a copy of the frame."""
        annotated = frame_info.frame.copy()

        for detection in frame_info.detections:
            self.draw_detection(annotated, detection)

        self.draw_hud(annotated, frame_info)

        if self.show_alert_banner and frame_info.has_detection:
            self.draw_alert_banner(annotated, frame_info.detection_count)

        return annotated

    def save(self, frame_or_info: Union[np.ndarray, FrameInfo], output_path: Union[str, Path]) -> bool:
        """Save frame or FrameInfo to disk."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(frame_or_info, FrameInfo):
            img = self.render(frame_or_info)
        else:
            img = frame_or_info

        return cv2.imwrite(str(target), img)

    def __call__(self, frame_info: FrameInfo) -> np.ndarray:
        return self.render(frame_info)
