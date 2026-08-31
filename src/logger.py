"""
Centralized Logging & CSV Detection Audit Logger
Provides rotating file and console logging, plus structured CSV logging for detection events.
"""

from __future__ import annotations
import csv
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Union

from src.core.frame_info import FrameInfo


class Logger:
    """
    Centralized application and detection event logger.
    """

    DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    CSV_HEADER = [
        "timestamp",
        "frame_id",
        "class_name",
        "confidence",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "fps",
        "latency_ms",
    ]

    def __init__(
        self,
        name: str = "elephant_detection",
        log_directory: Union[str, Path] = "logs",
        log_file: str = "app.log",
        csv_file: str = "detections.csv",
        level: str = "INFO",
        console: bool = True,
        file_logging: bool = True,
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 3,
    ) -> None:
        self.name = name
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.app_log_path = self.log_directory / log_file
        self.csv_log_path = self.log_directory / csv_file

        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self.logger.handlers.clear()

        formatter = logging.Formatter(self.DEFAULT_FORMAT, self.DATE_FORMAT)

        if console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        if file_logging:
            file_handler = RotatingFileHandler(
                str(self.app_log_path),
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        self._ensure_csv_header()

    def _ensure_csv_header(self) -> None:
        """Create CSV file with headers if it doesn't already exist or is empty."""
        if not self.csv_log_path.exists() or self.csv_log_path.stat().st_size == 0:
            try:
                with open(self.csv_log_path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.CSV_HEADER)
            except Exception:
                pass

    def log_frame_detections(self, frame_info: FrameInfo) -> None:
        """Log all detections in a frame to the CSV audit file."""
        if not frame_info.has_detection:
            return

        ts = frame_info.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.csv_log_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for det in frame_info.detections:
                    writer.writerow([
                        ts,
                        frame_info.frame_id,
                        det.class_name,
                        f"{det.confidence:.4f}",
                        det.x1,
                        det.y1,
                        det.x2,
                        det.y2,
                        f"{frame_info.fps:.2f}",
                        f"{frame_info.inference_time:.2f}",
                    ])
        except Exception as exc:
            self.logger.error("Failed to write to CSV log: %s", exc)

    def info(self, msg: str, *args) -> None:
        self.logger.info(msg, *args)

    def debug(self, msg: str, *args) -> None:
        self.logger.debug(msg, *args)

    def warning(self, msg: str, *args) -> None:
        self.logger.warning(msg, *args)

    def error(self, msg: str, *args) -> None:
        self.logger.error(msg, *args)

    def exception(self, msg: str, *args) -> None:
        self.logger.exception(msg, *args)


_DEFAULT_LOGGER: Optional[Logger] = None


def get_logger(name: str = "elephant_detection") -> Logger:
    """Return singleton application logger."""
    global _DEFAULT_LOGGER
    if _DEFAULT_LOGGER is None:
        _DEFAULT_LOGGER = Logger(name=name)
    return _DEFAULT_LOGGER