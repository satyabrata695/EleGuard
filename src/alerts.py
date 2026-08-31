"""
Alert Management System
Handles elephant detection alert generation, cooldown suppression, audio notification, and callback hooks.
"""

from __future__ import annotations
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from src.core.detection import Detection
from src.core.frame_info import FrameInfo


@dataclass(slots=True)
class Alert:
    """Represents a generated detection alert."""

    alert_type: str
    message: str
    confidence: float
    frame_id: int
    timestamp: datetime = field(default_factory=datetime.now)
    detection: Optional[Detection] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_type": self.alert_type,
            "message": self.message,
            "confidence": round(self.confidence, 4),
            "frame_id": self.frame_id,
            "timestamp": self.timestamp.isoformat(),
            "detection": self.detection.to_dict() if self.detection else None,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        return f"[{self.alert_type}] {self.message} (conf={self.confidence:.1%})"


class AlertManager:
    """
    Manages detection alerts, audio alarms, cooldowns, and callback notifications.
    """

    DEFAULT_ALERT_TYPE = "ELEPHANT_DETECTED"

    def __init__(
        self,
        confidence_threshold: float = 0.50,
        cooldown_seconds: float = 10.0,
        target_class: str = "elephant",
        console_alert: bool = True,
        sound_alert: bool = True,
        max_history: int = 500,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.target_class = target_class.strip().lower()
        self.console_alert = console_alert
        self.sound_alert = sound_alert
        self.max_history = max(10, int(max_history))

        self.alert_history: List[Alert] = []
        self._callbacks: List[Callable[[Alert], None]] = []
        self._last_alert_time: Optional[float] = None
        self._total_alerts = 0
        self._suppressed_alerts = 0

    def _is_target(self, detection: Detection) -> bool:
        return detection.class_name.strip().lower() == self.target_class

    def _cooldown_active(self) -> bool:
        if self._last_alert_time is None:
            return False
        return (time.monotonic() - self._last_alert_time) < self.cooldown_seconds

    def remaining_cooldown(self) -> float:
        if self._last_alert_time is None:
            return 0.0
        elapsed = time.monotonic() - self._last_alert_time
        return max(0.0, self.cooldown_seconds - elapsed)

    def _play_sound(self) -> None:
        """Play alert sound in a separate non-blocking thread."""
        def _sound_worker():
            try:
                if sys.platform.startswith("win"):
                    import winsound
                    winsound.Beep(1200, 350)
                else:
                    sys.stdout.write("\a")
                    sys.stdout.flush()
            except Exception:
                pass

        thread = threading.Thread(target=_sound_worker, daemon=True)
        thread.start()

    def trigger(self, alert: Alert) -> bool:
        """
        Trigger an alert if cooldown has expired.
        """
        if self._cooldown_active():
            self._suppressed_alerts += 1
            return False

        self._last_alert_time = time.monotonic()
        self._total_alerts += 1
        self.alert_history.append(alert)

        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)

        if self.console_alert:
            ts = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n🚨 [ALERT {ts}] {alert.message} (Frame: {alert.frame_id}, Conf: {alert.confidence:.1%})")

        if self.sound_alert:
            self._play_sound()

        for cb in self._callbacks:
            try:
                cb(alert)
            except Exception:
                pass

        return True

    def process(self, frame_info: FrameInfo) -> List[Alert]:
        """
        Process a FrameInfo object and trigger alerts for detected targets.
        """
        if frame_info is None or not frame_info.has_detection:
            return []

        target_dets = [
            d for d in frame_info.detections
            if self._is_target(d) and d.confidence >= self.confidence_threshold
        ]

        if not target_dets:
            return []

        top_det = max(target_dets, key=lambda d: d.confidence)
        alert = Alert(
            alert_type=self.DEFAULT_ALERT_TYPE,
            message=f"Elephant detected with {top_det.confidence:.1%} confidence.",
            confidence=top_det.confidence,
            frame_id=frame_info.frame_id,
            detection=top_det,
            metadata={"count": len(target_dets), "fps": frame_info.fps, "latency": frame_info.inference_time},
        )

        return [alert] if self.trigger(alert) else []

    def add_callback(self, callback: Callable[[Alert], None]) -> None:
        if callable(callback) and callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[Alert], None]) -> bool:
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            return True
        return False

    def clear_callbacks(self) -> None:
        self._callbacks.clear()

    def get_history(self) -> List[Alert]:
        return list(self.alert_history)

    def statistics(self) -> Dict[str, Any]:
        return {
            "total_alerts": self._total_alerts,
            "suppressed_alerts": self._suppressed_alerts,
            "cooldown_remaining": round(self.remaining_cooldown(), 1),
            "history_count": len(self.alert_history),
        }