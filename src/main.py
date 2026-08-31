"""
RT-DETR Elephant Detection System - Main Application Entry Point
Coordinates Camera acquisition, RT-DETR inference, Visualization HUD, Alert management, and Logging.

Usage Examples:
    python src/main.py --source 0
    python src/main.py --source "data/images/elephant1.png" --save
    python src/main.py --source "data/videos/wildlife.mp4" --confidence 0.45
    python src/main.py --source "rtsp://..." --no-display
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
from typing import Optional, Union
import cv2

from src.alerts import AlertManager
from src.camera import Camera
from src.detector import ElephantDetector
from src.logger import get_logger
from src.utils import FPSCounter, create_project_directories, generate_filename
from src.visualize import Visualizer

WINDOW_NAME = "RT-DETR Elephant Detection System"


class ElephantDetectionApp:
    """Main application orchestrator."""

    def __init__(
        self,
        model_path: Union[str, Path] = "weights/rtdetr-l.pt",
        source: Union[int, str, Path] = 0,
        confidence: float = 0.50,
        image_size: int = 640,
        device: Optional[str] = None,
        save_output: bool = False,
        output_directory: Union[str, Path] = "outputs/images",
        show_window: bool = True,
        alert_cooldown: float = 10.0,
        sound_alert: bool = True,
        warmup: bool = True,
    ) -> None:
        self.logger = get_logger("elephant_detection")
        self.model_path = Path(model_path)
        self.source = source
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.save_output = save_output
        self.output_directory = Path(output_directory)
        self.show_window = show_window
        self.alert_cooldown = alert_cooldown
        self.sound_alert = sound_alert
        self.warmup = warmup

        self.camera: Optional[Camera] = None
        self.detector: Optional[ElephantDetector] = None
        self.visualizer: Optional[Visualizer] = None
        self.alert_manager: Optional[AlertManager] = None
        self.fps_counter = FPSCounter()
        self.running = False
        self.frame_id = 0
        self.saved_images = 0

    def initialize(self) -> None:
        """Initialize all pipeline components."""
        self.logger.info("Initializing RT-DETR Elephant Detection System...")
        create_project_directories()

        # Check weights path fallback
        if not self.model_path.exists():
            root = Path(__file__).resolve().parent.parent
            if (root / "weights" / "best.pt").exists():
                self.model_path = root / "weights" / "best.pt"
            elif (root / "weights" / "rtdetr-l.pt").exists():
                self.model_path = root / "weights" / "rtdetr-l.pt"

        self.logger.info("Loading model weights: %s", self.model_path)
        self.detector = ElephantDetector(
            model_path=self.model_path,
            confidence=self.confidence,
            image_size=self.image_size,
            device=self.device,
            warmup=self.warmup,
        )

        self.camera = Camera(source=self.source, reconnect=True)
        if not self.camera.open():
            raise RuntimeError(f"Could not open input source: {self.source}")

        self.visualizer = Visualizer(show_fps=True, show_confidence=True, show_alert_banner=True)
        self.alert_manager = AlertManager(
            confidence_threshold=self.confidence,
            cooldown_seconds=self.alert_cooldown,
            sound_alert=self.sound_alert,
        )

        if self.save_output:
            self.output_directory.mkdir(parents=True, exist_ok=True)

        self.logger.info("Application initialized successfully. Source resolution: %dx%d", self.camera.width, self.camera.height)

    def run(self) -> None:
        """Run the detection loop."""
        if not self.camera or not self.detector or not self.visualizer or not self.alert_manager:
            raise RuntimeError("Application is not initialized.")

        self.running = True
        self.logger.info("Starting detection stream. Press 'q' or ESC in display window to exit.")
        is_static_image = getattr(self.camera, "_is_image_file", False)

        try:
            while self.running:
                success, frame = self.camera.read()
                if not success:
                    if self.camera.is_open:
                        continue
                    break

                current_fps = self.fps_counter.update()
                frame_info = self.detector.detect(frame=frame, frame_id=self.frame_id, fps=current_fps)

                # Process alerts & log detection audit
                alerts = self.alert_manager.process(frame_info)
                if frame_info.has_detection:
                    self.logger.log_frame_detections(frame_info)

                # Render annotations
                annotated_frame = self.visualizer.render(frame_info)

                # Save output if requested
                if self.save_output and (frame_info.has_detection or is_static_image):
                    out_name = generate_filename(prefix="elephant_det", extension=".jpg")
                    out_path = self.output_directory / out_name
                    cv2.imwrite(str(out_path), annotated_frame)
                    self.saved_images += 1
                    self.logger.info("Saved detection frame to: %s", out_path)

                # Display frame
                if self.show_window:
                    cv2.imshow(WINDOW_NAME, annotated_frame)
                    wait_time = 0 if is_static_image else 1
                    key = cv2.waitKey(wait_time) & 0xFF
                    if key in (ord("q"), 27):  # 'q' or ESC
                        break
                    elif key == ord("s"):  # 's' to manually snapshot
                        snap_name = generate_filename(prefix="snapshot", extension=".jpg")
                        cv2.imwrite(str(self.output_directory / snap_name), annotated_frame)
                        self.logger.info("Manual snapshot saved: %s", snap_name)

                self.frame_id += 1
                if is_static_image:
                    break

        except KeyboardInterrupt:
            self.logger.info("Detection interrupted by user.")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Cleanly release resources."""
        self.running = False
        if self.camera:
            self.camera.close()
        if self.detector:
            self.detector.release()
        cv2.destroyAllWindows()
        self.logger.info("Pipeline stopped. Total frames: %d, Saved images: %d", self.frame_id, self.saved_images)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RT-DETR Elephant Detection System")
    parser.add_argument("--source", type=str, default="0", help="Camera index, image file, video file, or RTSP stream")
    parser.add_argument("--model", type=str, default="weights/rtdetr-l.pt", help="Path to model weights")
    parser.add_argument("--confidence", type=float, default=0.50, help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", type=str, default=None, help="Execution device: auto, cuda, cpu")
    parser.add_argument("--save", action="store_true", help="Save frames with detected elephants")
    parser.add_argument("--output", type=str, default="outputs/images", help="Output directory for saved frames")
    parser.add_argument("--no-display", action="store_true", help="Run in headless mode without GUI window")
    parser.add_argument("--cooldown", type=float, default=10.0, help="Alert cooldown in seconds")
    parser.add_argument("--no-sound", action="store_true", help="Disable audio sound alerts")
    return parser


def main() -> int:
    parser = create_argument_parser() if "create_argument_parser" in globals() else create_parser()
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    app = ElephantDetectionApp(
        model_path=args.model,
        source=src,
        confidence=args.confidence,
        image_size=args.imgsz,
        device=args.device,
        save_output=args.save,
        output_directory=args.output,
        show_window=not args.no_display,
        alert_cooldown=args.cooldown,
        sound_alert=not args.no_sound,
    )
    app.initialize()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())