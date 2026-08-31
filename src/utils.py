"""
Project Utility Functions & Telemetry Helpers
Provides directory creation, timestamp generation, FPS estimation, and system diagnostics.
"""

from __future__ import annotations
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch


def ensure_directory(directory: Union[str, Path]) -> Path:
    """Create directory if not present."""
    p = Path(directory)
    p.mkdir(parents=True, exist_ok=True)
    return p


def create_project_directories(base_dir: Optional[Union[str, Path]] = None) -> List[Path]:
    """Ensure standard project folders exist."""
    root = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent
    dirs = [
        root / "data" / "images",
        root / "data" / "videos",
        root / "data" / "test",
        root / "outputs" / "images",
        root / "outputs" / "videos",
        root / "logs",
        root / "weights",
        root / "runs",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def current_timestamp() -> datetime:
    return datetime.now()


def timestamp_string(include_microseconds: bool = False) -> str:
    fmt = "%Y%m%d_%H%M%S_%f" if include_microseconds else "%Y%m%d_%H%M%S"
    return datetime.now().strftime(fmt)


def generate_filename(prefix: str = "elephant", extension: str = ".jpg", include_microseconds: bool = False) -> str:
    """Generate a clean timestamped filename."""
    ext = extension if extension.startswith(".") else f".{extension}"
    return f"{prefix}_{timestamp_string(include_microseconds)}{ext}"


class FPSCounter:
    """
    Moving average FPS counter with exponential smoothing.
    """

    def __init__(self, smoothing: float = 0.9) -> None:
        self.smoothing = max(0.0, min(1.0, float(smoothing)))
        self._fps = 0.0
        self._last_time = time.perf_counter()
        self._frame_count = 0

    def update(self) -> float:
        """Call on each processed frame to compute instant and smoothed FPS."""
        now = time.perf_counter()
        elapsed = now - self._last_time
        self._last_time = now
        self._frame_count += 1

        if elapsed > 0:
            instant_fps = 1.0 / elapsed
            self._fps = instant_fps if self._fps == 0.0 else (self.smoothing * self._fps + (1.0 - self.smoothing) * instant_fps)
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._frame_count

    def reset(self) -> None:
        self._fps = 0.0
        self._last_time = time.perf_counter()
        self._frame_count = 0


def get_system_info() -> Dict[str, Any]:
    """Return platform, Python, and GPU diagnostic summary."""
    gpu_info = "Not available"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        gpu_info = f"{name} ({vram_gb:.2f} GB VRAM, CUDA {torch.version.cuda})"

    return {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu_info,
    }