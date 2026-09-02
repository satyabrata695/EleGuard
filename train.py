"""
RT-DETR Elephant Detection System - Training Controller
Fine-tunes the RT-DETR model on custom dataset with validation, checkpointing, and GPU optimization.

Usage Examples:
    python train.py --epochs 25 --batch 4 --imgsz 640
    python train.py --data "data/data.yaml" --model "weights/rtdetr-l.pt" --device cuda
"""

from __future__ import annotations
import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, Union
import torch
from ultralytics import RTDETR

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "data" / "data.yaml"
PRETRAINED_MODEL = PROJECT_ROOT / "weights" / "rtdetr-l.pt"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
RUNS_DIR = PROJECT_ROOT / "runs"


class ElephantRTDETRTrainer:
    """Production RT-DETR model training controller."""

    def __init__(
        self,
        model_path: Union[str, Path] = PRETRAINED_MODEL,
        data_yaml: Union[str, Path] = DATA_YAML,
        epochs: int = 25,
        image_size: int = 640,
        batch_size: int = 4,
        workers: int = 2,
        patience: int = 15,
        device: str = "auto",
        run_name: str = "elephant_rtdetr",
        seed: int = 42,
        resume: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.data_yaml = Path(data_yaml)
        self.epochs = max(1, int(epochs))
        self.image_size = max(32, int(image_size))
        self.batch_size = max(1, int(batch_size))
        self.workers = max(0, int(workers))
        self.patience = max(1, int(patience))
        self.device = self._resolve_device(device)
        self.run_name = run_name.strip() or "elephant_rtdetr"
        self.seed = int(seed)
        self.resume = bool(resume)

        self.model: Optional[RTDETR] = None
        self.start_time = 0.0
        self._validate_paths()

    @staticmethod
    def _resolve_device(req_device: str) -> str:
        req = str(req_device).strip().lower()
        if req == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if req.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA-capable GPU is available.")
        return req

    def _validate_paths(self) -> None:
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"Dataset YAML file not found: {self.data_yaml}")
        if not self.model_path.exists() and self.model_path.name not in ("rtdetr-l.pt", "rtdetr-x.pt"):
            raise FileNotFoundError(f"Pretrained model weights not found: {self.model_path}")

    def validate_dataset(self) -> None:
        """Verify train and validation image directories."""
        dataset_root = self.data_yaml.parent
        train_dir = dataset_root / "train" / "images"
        valid_dir = dataset_root / "valid" / "images"

        print("\n--- Dataset Validation ---")
        train_count = len(list(train_dir.glob("*"))) if train_dir.exists() else 0
        valid_count = len(list(valid_dir.glob("*"))) if valid_dir.exists() else 0
        print(f"Training images   : {train_count} in {train_dir}")
        print(f"Validation images : {valid_count} in {valid_dir}")

        if train_count == 0:
            raise RuntimeError(f"No training images found in: {train_dir}")

    def print_environment(self) -> None:
        """Print training hardware & software summary."""
        print("=" * 60)
        print("RT-DETR ELEPHANT DETECTION - TRAINING CONTROLLER")
        print("=" * 60)
        print(f"Python      : {sys.version.split()[0]}")
        print(f"PyTorch     : {torch.__version__}")
        print(f"Device      : {self.device}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"GPU         : {gpu_name} ({gpu_mem:.2f} GB)")
        print(f"Model       : {self.model_path}")
        print(f"Data YAML   : {self.data_yaml}")
        print(f"Epochs      : {self.epochs} | Batch Size: {self.batch_size} | ImgSz: {self.image_size}")
        print("=" * 60)

    def train(self) -> int:
        """Execute RT-DETR training pipeline."""
        self.print_environment()
        self.validate_dataset()

        print("\nLoading RT-DETR model...")
        model_src = str(self.model_path) if self.model_path.exists() else self.model_path.name
        self.model = RTDETR(model_src)

        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

        print("\nStarting model training...")
        self.start_time = time.perf_counter()

        results = self.model.train(
            data=str(self.data_yaml),
            epochs=self.epochs,
            imgsz=self.image_size,
            batch=self.batch_size,
            device=self.device,
            workers=self.workers,
            patience=self.patience,
            project=str(RUNS_DIR),
            name=self.run_name,
            exist_ok=True,
            pretrained=True,
            amp=True,
            seed=self.seed,
            resume=self.resume,
            save=True,
            val=True,
            plots=True,
            verbose=True,
        )

        elapsed_mins = (time.perf_counter() - self.start_time) / 60.0
        print(f"\nTraining completed in {elapsed_mins:.2f} minutes.")

        # Locate and copy best model
        best_pt = RUNS_DIR / self.run_name / "weights" / "best.pt"
        if best_pt.exists():
            dest = WEIGHTS_DIR / "best.pt"
            shutil.copy2(best_pt, dest)
            print(f"✓ Best model saved to: {dest}")
        else:
            print(f"Check run output directory: {RUNS_DIR / self.run_name}")

        return 0


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train RT-DETR Elephant Detection Model")
    parser.add_argument("--model", type=str, default=str(PRETRAINED_MODEL), help="Path to initial weights")
    parser.add_argument("--data", type=str, default=str(DATA_YAML), help="Path to dataset data.yaml")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (e.g. 4 or 8 for Laptop GPUs)")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader worker threads")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, cpu")
    parser.add_argument("--name", type=str, default="elephant_rtdetr", help="Training run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    trainer = ElephantRTDETRTrainer(
        model_path=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        image_size=args.imgsz,
        batch_size=args.batch,
        workers=args.workers,
        patience=args.patience,
        device=args.device,
        run_name=args.name,
        seed=args.seed,
        resume=args.resume,
    )
    try:
        return trainer.train()
    except KeyboardInterrupt:
        print("\nTraining aborted by user.")
        return 130
    except Exception as exc:
        print(f"\nTraining Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())