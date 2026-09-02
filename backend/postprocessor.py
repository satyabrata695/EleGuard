"""
Inference Post-Processing Module
Converts raw RT-DETR/Ultralytics results into structured Detection objects.
Includes confidence filtering, class filtering, boundary clipping, and IoU duplicate suppression.
"""

from __future__ import annotations
from typing import Any, Iterable, List, Optional, Set
import numpy as np

from backend.core.detection import Detection


class PostProcessor:
    """
    Processes raw RT-DETR prediction results into Detection objects.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        target_classes: Optional[Iterable[str]] = None,
        clip_boxes: bool = True,
        sort_by_confidence: bool = True,
        max_detections: Optional[int] = None,
        duplicate_iou_threshold: float = 0.70,
        enable_duplicate_suppression: bool = True,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0.")
        if not 0.0 <= duplicate_iou_threshold <= 1.0:
            raise ValueError("duplicate_iou_threshold must be between 0.0 and 1.0.")
        if max_detections is not None and max_detections <= 0:
            raise ValueError("max_detections must be greater than zero.")

        self.confidence_threshold = float(confidence_threshold)
        self.target_classes = self._normalize_target_classes(target_classes)
        self.clip_boxes = bool(clip_boxes)
        self.sort_by_confidence = bool(sort_by_confidence)
        self.max_detections = max_detections
        self.duplicate_iou_threshold = float(duplicate_iou_threshold)
        self.enable_duplicate_suppression = bool(enable_duplicate_suppression)

        self._processed_frames = 0
        self._total_detections = 0

    @staticmethod
    def _normalize_target_classes(target_classes: Optional[Iterable[str]]) -> Optional[Set[str]]:
        if target_classes is None:
            return None
        return {str(cls_name).strip().lower() for cls_name in target_classes if str(cls_name).strip()}

    @staticmethod
    def calculate_iou(box_a: tuple, box_b: tuple) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
        x_left = max(box_a[0], box_b[0])
        y_top = max(box_a[1], box_b[1])
        x_right = min(box_a[2], box_b[2])
        y_bottom = min(box_a[3], box_b[3])

        intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
        if intersection_area == 0:
            return 0.0

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union_area = float(area_a + area_b - intersection_area)

        return intersection_area / union_area if union_area > 0 else 0.0

    def suppress_duplicates(self, detections: List[Detection]) -> List[Detection]:
        """Suppress overlapping detections of the same class based on IoU threshold."""
        if not detections or len(detections) <= 1:
            return detections

        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: List[Detection] = []

        for candidate in sorted_dets:
            should_keep = True
            for existing in kept:
                if existing.class_name.lower() == candidate.class_name.lower():
                    iou = self.calculate_iou(existing.bbox, candidate.bbox)
                    if iou > self.duplicate_iou_threshold:
                        should_keep = False
                        break
            if should_keep:
                kept.append(candidate)

        return kept

    def process(self, results: Any, image_shape: Optional[tuple] = None) -> List[Detection]:
        """
        Convert Ultralytics Results object into a list of Detection objects.
        """
        self._processed_frames += 1
        if results is None or not hasattr(results, "boxes") or results.boxes is None:
            return []

        boxes = results.boxes
        if len(boxes) == 0:
            return []

        # Extract numpy or torch tensors
        xyxy_arr = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
        conf_arr = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
        cls_arr = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
        names = getattr(results, "names", {})

        img_height, img_width = (None, None)
        if image_shape is not None:
            img_height, img_width = image_shape[:2]
        elif hasattr(results, "orig_shape") and results.orig_shape is not None:
            img_height, img_width = results.orig_shape[:2]

        detections: List[Detection] = []

        for i in range(len(conf_arr)):
            confidence = float(conf_arr[i])
            if confidence < self.confidence_threshold:
                continue

            class_id = int(cls_arr[i])
            class_name = names.get(class_id, str(class_id))

            if self.target_classes is not None:
                if class_name.lower() not in self.target_classes:
                    continue

            x1, y1, x2, y2 = xyxy_arr[i]
            x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

            if self.clip_boxes and img_width is not None and img_height is not None:
                x1 = max(0, min(x1, img_width - 1))
                y1 = max(0, min(y1, img_height - 1))
                x2 = max(0, min(x2, img_width - 1))
                y2 = max(0, min(y2, img_height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                )
            )

        if self.enable_duplicate_suppression:
            detections = self.suppress_duplicates(detections)

        if self.sort_by_confidence:
            detections.sort(key=lambda d: d.confidence, reverse=True)

        if self.max_detections is not None and len(detections) > self.max_detections:
            detections = detections[: self.max_detections]

        self._total_detections += len(detections)
        return detections

    def __call__(self, results: Any, image_shape: Optional[tuple] = None) -> List[Detection]:
        return self.process(results, image_shape)
