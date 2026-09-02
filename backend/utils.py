"""
Utility & System Diagnostics Module
Provides FPS calculation, hardware profiling, and directory setup.
"""

from __future__ import annotations
import platform
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict


class FPSCounter:
    """Moving-average FPS counter."""

    def __init__(self, window_size: int = 30) -> None:
        self.window_size = window_size
        self._timestamps: deque[float] = deque(maxlen=window_size)
        self._last_time: float = 0.0

    def tick(self) -> float:
        """Call once per processed frame. Returns current moving-average FPS."""
        now = time.perf_counter()
        if self._last_time > 0.0:
            dt = now - self._last_time
            if dt > 0:
                self._timestamps.append(dt)
        self._last_time = now

        if not self._timestamps:
            return 0.0
        avg_dt = sum(self._timestamps) / len(self._timestamps)
        return 1.0 / avg_dt if avg_dt > 0.0 else 0.0

    @property
    def fps(self) -> float:
        if not self._timestamps:
            return 0.0
        avg_dt = sum(self._timestamps) / len(self._timestamps)
        return 1.0 / avg_dt if avg_dt > 0.0 else 0.0

    def reset(self) -> None:
        self._timestamps.clear()
        self._last_time = 0.0


def create_project_directories() -> None:
    """Ensure all expected runtime directories exist."""
    root = Path(__file__).resolve().parent.parent
    for dir_name in ("data", "data/images", "data/videos", "logs", "weights", "outputs/images", "outputs/videos", "runs"):
        (root / dir_name).mkdir(parents=True, exist_ok=True)


def generate_filename(prefix: str = "elephant_det", extension: str = "jpg") -> str:
    """Generate timestamped output filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = extension.lstrip(".")
    return f"{prefix}_{timestamp}.{ext}"


def get_system_info() -> Dict[str, str]:
    """Gather hardware and environment telemetry."""
    import torch
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "device_count": str(torch.cuda.device_count()) if torch.cuda.is_available() else "0",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU Mode)",
    }
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        info["vram"] = f"{vram_gb:.2f} GB"
    return info
