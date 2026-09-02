"""
RT-DETR Elephant Detector Engine
Handles RT-DETR model loading, device management, warm-up, and real-time frame inference.
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import numpy as np
import torch
from ultralytics import RTDETR

from backend.core.frame_info import FrameInfo
from backend.postprocessor import PostProcessor


class ElephantDetector:
    """
    High-performance RT-DETR Elephant Detector.
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.50,
        image_size: int = 640,
        device: Optional[str] = None,
        iou: float = 0.45,
        max_detections: int = 300,
        target_classes: Optional[Iterable[str]] = ("elephant",),
        warmup: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.iou = float(iou)
        self.max_detections = int(max_detections)
        self.device = self._select_device(device)

        self.model = self._load_model()
        self.postprocessor = PostProcessor(
            confidence_threshold=self.confidence,
            target_classes=target_classes,
            duplicate_iou_threshold=self.iou,
            max_detections=self.max_detections,
        )

        self._warmup_complete = False
        self._last_inference_time_ms = 0.0
        self._total_inferences = 0
        self._total_inference_time_ms = 0.0

        if warmup:
            self.warmup()

    @staticmethod
    def _select_device(device: Optional[str]) -> str:
        """Select execution device ('0', 'cuda', 'cpu', or specific cuda index)."""
        if device is None or not str(device).strip():
            return "0" if torch.cuda.is_available() else "cpu"

        device_clean = str(device).strip().lower()
        if device_clean in ("auto", "none", "default"):
            return "0" if torch.cuda.is_available() else "cpu"

        if device_clean == "cuda":
            return "0" if torch.cuda.is_available() else "cpu"

        if device_clean.startswith("cuda:"):
            if not torch.cuda.is_available():
                return "cpu"
            return device_clean.replace("cuda:", "")

        if device_clean.isdigit():
            if not torch.cuda.is_available():
                return "cpu"
            return device_clean

        if device_clean.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"

        return device_clean

    def _load_model(self) -> RTDETR:
        """Load RT-DETR model weights, auto-downloading base model if needed."""
        if self.model_path.exists():
            return RTDETR(str(self.model_path))
        if self.model_path.name in ("rtdetr-l.pt", "rtdetr-x.pt", "yolo26n.pt", "yolo11n.pt"):
            return RTDETR(self.model_path.name)
        # Automatic fallback to standard downloadable RT-DETR checkpoint if file missing on server
        return RTDETR("rtdetr-l.pt")

    def warmup(self) -> None:
        """Warm up the model with a dummy inference to eliminate cold-start latency."""
        if self._warmup_complete:
            return
        dummy = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        try:
            self.model.predict(
                source=dummy,
                imgsz=self.image_size,
                conf=self.confidence,
                iou=self.iou,
                device=self.device,
                verbose=False,
            )
            self._warmup_complete = True
        except Exception:
            pass

    def detect(self, frame: np.ndarray, frame_id: int = 0, fps: float = 0.0) -> FrameInfo:
        """
        Execute RT-DETR detection on a single frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from OpenCV.
        frame_id : int
            Sequential frame index.
        fps : float
            Current pipeline FPS.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame supplied for detection.")

        start_time = time.perf_counter()

        try:
            results = self.model.predict(
                source=frame,
                imgsz=self.image_size,
                conf=self.confidence,
                iou=self.iou,
                device=self.device,
                max_det=self.max_detections,
                verbose=False,
            )
        except Exception:
            # Automatic fallback to CPU if device inference fails
            results = self.model.predict(
                source=frame,
                imgsz=self.image_size,
                conf=self.confidence,
                iou=self.iou,
                device="cpu",
                max_det=self.max_detections,
                verbose=False,
            )

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0
        detections = self.postprocessor(results[0], image_shape=frame.shape) if results else []

        self._last_inference_time_ms = inference_time_ms
        self._total_inferences += 1
        self._total_inference_time_ms += inference_time_ms

        return FrameInfo(
            frame_id=frame_id,
            frame=frame,
            detections=detections,
            fps=fps,
            inference_time=inference_time_ms,
        )

    def detect_batch(self, frames: List[np.ndarray], start_frame_id: int = 0) -> List[FrameInfo]:
        """Execute detection on a batch of frames."""
        return [self.detect(f, frame_id=start_frame_id + i) for i, f in enumerate(frames)]

    def set_confidence(self, confidence: float) -> None:
        """Update detection confidence threshold dynamically."""
        self.confidence = max(0.0, min(1.0, float(confidence)))
        self.postprocessor.confidence_threshold = self.confidence

    @property
    def class_names(self) -> Dict[int, str]:
        return getattr(self.model, "names", {})

    @property
    def number_of_classes(self) -> int:
        return len(self.class_names)

    @property
    def using_gpu(self) -> bool:
        return self.device != "cpu" and torch.cuda.is_available()

    @property
    def last_inference_time_ms(self) -> float:
        return self._last_inference_time_ms

    @property
    def average_inference_time_ms(self) -> float:
        return self._total_inference_time_ms / self._total_inferences if self._total_inferences > 0 else 0.0

    @property
    def processed_frames(self) -> int:
        return self._total_inferences

    @property
    def average_fps(self) -> float:
        avg_ms = self.average_inference_time_ms
        return 1000.0 / avg_ms if avg_ms > 0 else 0.0

    def statistics(self) -> Dict[str, Any]:
        """Return runtime statistics."""
        return {
            "processed_frames": self.processed_frames,
            "last_inference_time_ms": round(self.last_inference_time_ms, 2),
            "average_inference_time_ms": round(self.average_inference_time_ms, 2),
            "average_fps": round(self.average_fps, 2),
            "device": self.device,
            "using_gpu": self.using_gpu,
        }

    def release(self) -> None:
        """Clear CUDA cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __call__(self, frame: np.ndarray, frame_id: int = 0, fps: float = 0.0) -> FrameInfo:
        return self.detect(frame, frame_id=frame_id, fps=fps)

    def __str__(self) -> str:
        return f"ElephantDetector(model={self.model_path.name}, device={self.device}, conf={self.confidence:.2f}, imgsz={self.image_size})"
