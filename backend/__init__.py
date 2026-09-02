"""
EleGuard Backend Package
AI Detection, Camera Streaming, Alert Management, and Telemetry Engine.
"""

from backend.alerts import Alert, AlertManager
from backend.camera import Camera
from backend.core.detection import Detection
from backend.core.frame_info import FrameInfo
from backend.detector import ElephantDetector
from backend.logger import Logger, get_logger
from backend.postprocessor import PostProcessor
from backend.utils import FPSCounter, get_system_info
from backend.visualize import Visualizer

__all__ = [
    "Alert",
    "AlertManager",
    "Camera",
    "Detection",
    "ElephantDetector",
    "FPSCounter",
    "FrameInfo",
    "Logger",
    "PostProcessor",
    "Visualizer",
    "get_logger",
    "get_system_info",
]
