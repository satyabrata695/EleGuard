"""
RT-DETR Elephant Detection System - Configuration Module
National Institute of Technology Rourkela
Department of ECE | 5G MEC Lab
"""

from pathlib import Path
from typing import Tuple

# =========================================================
# Project Information
# =========================================================
PROJECT_NAME = "RT-DETR Elephant Detection System"
VERSION = "2.5"
AUTHOR = "Satya"

# =========================================================
# File Paths & Directories
# =========================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
VIDEO_DIR = DATA_DIR / "videos"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_VIDEO_DIR = OUTPUT_DIR / "videos"
LOG_DIR = ROOT_DIR / "logs"
WEIGHTS_DIR = ROOT_DIR / "weights"
NOTEBOOK_DIR = ROOT_DIR / "notebooks"
RUN_DIR = ROOT_DIR / "runs"

# Ensure all required directories exist
DIRECTORIES = [
    IMAGE_DIR,
    VIDEO_DIR,
    TEST_DIR,
    OUTPUT_IMAGE_DIR,
    OUTPUT_VIDEO_DIR,
    LOG_DIR,
    WEIGHTS_DIR,
    NOTEBOOK_DIR,
    RUN_DIR,
]
for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)

# =========================================================
# Model & Inference Settings
# =========================================================
MODEL_NAME = "RT-DETR-Elephant"
MODEL_PATH = WEIGHTS_DIR / "best.pt" if (WEIGHTS_DIR / "best.pt").exists() else WEIGHTS_DIR / "rtdetr-l.pt"
DEVICE = "auto"  # "auto", "cuda", "cpu"
CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640
TARGET_CLASS = "elephant"

# =========================================================
# Input & Camera Settings
# =========================================================
INPUT_SOURCE = 0  # 0 for webcam, "video.mp4", "image.jpg", or "rtsp://..."
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# =========================================================
# Visualization Settings
# =========================================================
WINDOW_NAME = "RT-DETR Elephant Detection"
SHOW_FPS = True
SHOW_CONFIDENCE = True
SHOW_CLASS_NAME = True
SHOW_TIMESTAMP = True
SHOW_DETECTION_COUNT = True
SHOW_INFERENCE_TIME = True
FONT_SCALE = 0.6
FONT_THICKNESS = 1
BOX_THICKNESS = 2

# Colors (BGR)
BOX_COLOR: Tuple[int, int, int] = (0, 255, 0)
TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)
BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)
ALERT_BANNER_COLOR: Tuple[int, int, int] = (0, 0, 220)

# =========================================================
# Logging & Alert Settings
# =========================================================
ENABLE_LOGGING = True
APP_LOG_FILE = LOG_DIR / "app.log"
CSV_LOG_FILE = LOG_DIR / "detections.csv"
SAVE_DETECTED_IMAGES = True
SAVE_DETECTED_VIDEO = False

ENABLE_ALERT = True
ALERT_SOUND = True
ALERT_COOLDOWN = 10.0  # seconds between alerts

# =========================================================
# Dashboard Settings
# =========================================================
ENABLE_DASHBOARD = True
DASHBOARD_TITLE = "RT-DETR Elephant Detection System"

# Supported Extensions
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv")