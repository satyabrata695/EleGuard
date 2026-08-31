"""
RT-DETR Elephant Detection System - Diagnostics & Benchmark Test Suite
Validates Environment, CUDA, Core Modules, Model Weights, Inference, Postprocessing, and Visualization.

Usage:
    python test.py
    python test.py --image "data/images/elephant1.png" --model "weights/best.pt"
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Union
import cv2
import numpy as np
import torch
import ultralytics

from src.alerts import AlertManager
from src.camera import Camera
from src.core.detection import Detection
from src.core.frame_info import FrameInfo
from src.detector import ElephantDetector
from src.logger import Logger, get_logger
from src.postprocessor import PostProcessor
from src.utils import FPSCounter
from src.visualize import Visualizer


class TestResult:
    def __init__(self, name: str, success: bool, message: str, elapsed: float = 0.0) -> None:
        self.name = name
        self.success = success
        self.message = message
        self.elapsed = elapsed


class DiagnosticSuite:
    """Comprehensive test runner for the RT-DETR Elephant Detection System."""

    def __init__(
        self,
        model_path: Union[str, Path] = "weights/rtdetr-l.pt",
        image_path: Optional[Union[str, Path]] = "data/images/elephant1.png",
        output_path: Union[str, Path] = "outputs/test_detection.jpg",
        confidence: float = 0.25,
        image_size: int = 640,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.image_path = Path(image_path) if image_path else None
        self.output_path = Path(output_path)
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.results: list[TestResult] = []
        self.detector: Optional[ElephantDetector] = None

    def _record(self, name: str, success: bool, message: str, elapsed: float = 0.0) -> bool:
        self.results.append(TestResult(name, success, message, elapsed))
        status = "[PASS]" if success else "[FAIL]"
        time_str = f" ({elapsed:.3f}s)" if elapsed > 0 else ""
        print(f"  {status} {name}: {message}{time_str}")
        return success

    def test_environment(self) -> bool:
        print("\n1. ENVIRONMENT & DEPENDENCY CHECK")
        start = time.perf_counter()
        try:
            info = f"Python {sys.version.split()[0]} | PyTorch {torch.__version__} | OpenCV {cv2.__version__} | Ultralytics {ultralytics.__version__}"
            return self._record("Python Packages", True, info, time.perf_counter() - start)
        except Exception as exc:
            return self._record("Python Packages", False, str(exc), time.perf_counter() - start)

    def test_cuda(self) -> bool:
        print("\n2. HARDWARE & CUDA ACCELERATION")
        start = time.perf_counter()
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return self._record("CUDA GPU", True, f"Detected {name} ({vram:.2f} GB VRAM, CUDA {torch.version.cuda})", time.perf_counter() - start)
        return self._record("CUDA GPU", True, "CUDA not detected. Running in CPU mode.", time.perf_counter() - start)

    def test_project_modules(self) -> bool:
        print("\n3. CORE PROJECT MODULES")
        start = time.perf_counter()
        try:
            _ = AlertManager(confidence_threshold=0.5, sound_alert=False)
            _ = Camera(source=0)
            _ = Visualizer()
            _ = PostProcessor()
            _ = Logger(name="test_log", console=False, file_logging=False)
            return self._record("Module Imports", True, "All project modules initialized successfully.", time.perf_counter() - start)
        except Exception as exc:
            return self._record("Module Imports", False, str(exc), time.perf_counter() - start)

    def test_model_loading(self) -> bool:
        print("\n4. MODEL LOADING & WARM-UP")
        start = time.perf_counter()
        # Fallback to base model if best.pt is specified but missing
        if not self.model_path.exists():
            root = Path(__file__).resolve().parent
            if (root / "weights" / "best.pt").exists():
                self.model_path = root / "weights" / "best.pt"
            elif (root / "weights" / "rtdetr-l.pt").exists():
                self.model_path = root / "weights" / "rtdetr-l.pt"

        try:
            self.detector = ElephantDetector(
                model_path=self.model_path,
                confidence=self.confidence,
                image_size=self.image_size,
                device=self.device,
                warmup=True,
            )
            return self._record(
                "RT-DETR Model",
                True,
                f"Loaded {self.model_path.name} on {self.detector.device} with {self.detector.number_of_classes} classes.",
                time.perf_counter() - start,
            )
        except Exception as exc:
            return self._record("RT-DETR Model", False, str(exc), time.perf_counter() - start)

    def test_inference(self) -> bool:
        print("\n5. DETECTION INFERENCE & POSTPROCESSING")
        if not self.detector:
            return self._record("Inference", False, "Detector was not initialized.")

        # Load image or create synthetic frame
        if self.image_path and self.image_path.exists():
            frame = cv2.imread(str(self.image_path))
            src_desc = f"image '{self.image_path.name}'"
        else:
            frame = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            src_desc = "synthetic test frame"

        start = time.perf_counter()
        try:
            frame_info = self.detector.detect(frame)
            elapsed = time.perf_counter() - start
            msg = f"Processed {src_desc} in {frame_info.inference_time:.1f}ms ({frame_info.detection_count} detections)"
            for i, d in enumerate(frame_info.detections[:5], 1):
                msg += f"\n       [{i}] {d.class_name}: {d.confidence:.1%} @ {d.bbox}"

            rec = self._record("Inference Engine", True, msg, elapsed)

            # Test Visualizer & Output
            viz = Visualizer()
            annotated = viz.render(frame_info)
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.output_path), annotated)
            self._record("Visualization & Saving", True, f"Annotated output saved to: {self.output_path}")
            return rec
        except Exception as exc:
            return self._record("Inference Engine", False, str(exc), time.perf_counter() - start)

    def run(self) -> int:
        print("=" * 60)
        print("RT-DETR ELEPHANT DETECTION - DIAGNOSTIC SUITE")
        print("=" * 60)

        self.test_environment()
        self.test_cuda()
        self.test_project_modules()
        self.test_model_loading()
        self.test_inference()

        total = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        failed = total - passed

        print("\n" + "=" * 60)
        print(f"SUMMARY: {passed}/{total} checks passed (Failed: {failed})")
        if self.detector and self.detector.processed_frames > 0:
            print(f"Average Latency: {self.detector.average_inference_time_ms:.1f}ms | Average FPS: {self.detector.average_fps:.1f}")
        print("=" * 60)

        return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnostic Test Suite for Elephant Detection")
    parser.add_argument("--model", type=str, default="weights/rtdetr-l.pt", help="Model weights path")
    parser.add_argument("--image", type=str, default="data/images/elephant1.png", help="Test image path")
    parser.add_argument("--output", type=str, default="outputs/test_detection.jpg", help="Output annotated image path")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default=None, help="Inference device")
    args = parser.parse_args()

    suite = DiagnosticSuite(
        model_path=args.model,
        image_path=args.image,
        output_path=args.output,
        confidence=args.confidence,
        image_size=args.imgsz,
        device=args.device,
    )
    return suite.run()


if __name__ == "__main__":
    raise SystemExit(main())