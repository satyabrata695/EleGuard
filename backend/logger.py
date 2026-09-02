"""
Logging & Audit System
Provides structured console and rotating file logging, plus CSV detection audit trails.
"""

from __future__ import annotations
import csv
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from backend.core.frame_info import FrameInfo

CSV_HEADER = ["timestamp", "frame_id", "detection_count", "classes", "confidences", "boxes", "inference_time_ms", "fps"]


class Logger:
    """
    Manages application runtime logging and CSV telemetry audits.
    """

    def __init__(
        self,
        name: str = "eleguard",
        log_dir: str | Path = "logs",
        log_level: int = logging.INFO,
        console: bool = True,
        file_logging: bool = True,
        csv_logging: bool = True,
    ) -> None:
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / "app.log"
        self.csv_file = self.log_dir / "detections.csv"

        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.propagate = False

        if not self.logger.handlers:
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            if console:
                ch = logging.StreamHandler()
                ch.setLevel(log_level)
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)

            if file_logging:
                fh = RotatingFileHandler(
                    self.log_file,
                    maxBytes=5 * 1024 * 1024,
                    backupCount=3,
                    encoding="utf-8",
                )
                fh.setLevel(log_level)
                fh.setFormatter(formatter)
                self.logger.addHandler(fh)

        self.csv_logging = csv_logging
        if self.csv_logging and not self.csv_file.exists():
            with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADER)

    def info(self, msg: str, *args) -> None:
        self.logger.info(msg, *args)

    def warning(self, msg: str, *args) -> None:
        self.logger.warning(msg, *args)

    def error(self, msg: str, *args) -> None:
        self.logger.error(msg, *args)

    def debug(self, msg: str, *args) -> None:
        self.logger.debug(msg, *args)

    def log_detection(self, frame_info: FrameInfo) -> None:
        """Append detection telemetry record to CSV audit log."""
        if not self.csv_logging or not frame_info.has_detection:
            return

        ts = frame_info.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        classes = ";".join(d.class_name for d in frame_info.detections)
        confidences = ";".join(f"{d.confidence:.3f}" for d in frame_info.detections)
        boxes = ";".join(f"{d.bbox}" for d in frame_info.detections)

        row = [
            ts,
            frame_info.frame_id,
            frame_info.detection_count,
            classes,
            confidences,
            boxes,
            f"{frame_info.inference_time:.1f}",
            f"{frame_info.fps:.1f}",
        ]

        try:
            with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as exc:
            self.logger.error("Failed to write detection to CSV audit file: %s", exc)


def get_logger(name: str = "eleguard") -> Logger:
    """Return a singleton-like Logger instance."""
    return Logger(name=name)
