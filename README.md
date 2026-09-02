# 🐘 EleGuard — AI-Powered Wildlife Early Warning & Surveillance System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-RT--DETR-green.svg)](https://docs.ultralytics.com/models/rtdetr/)
[![Streamlit](https://img.shields.io/badge/Streamlit-EleGuard%20Dashboard-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**EleGuard** is an AI-powered, real-time Elephant Detection and Early Warning Surveillance System utilizing **RT-DETR (Real-Time Detection Transformer)**. Designed for wildlife preservation, human-elephant conflict mitigation, and smart railway/perimeter surveillance.

Developed at **5G MEC Lab, Department of ECE, National Institute of Technology Rourkela**.

---

## 📌 System Architecture

```mermaid
flowchart LR
    A[Input Source\nWebcam / Video / RTSP / Image] --> B[Camera Module\nsrc/camera.py]
    B --> C[RT-DETR Detector\nsrc/detector.py]
    C --> D[PostProcessor\nTarget Filter: Elephant\nsrc/postprocessor.py]
    D --> E[Visualizer & HUD\nsrc/visualize.py]
    D --> F[Alert Manager\nAudio Siren & Cooldown\nsrc/alerts.py]
    D --> G[Audit Logger\nCSV & App Logs\nsrc/logger.py]
    E --> H[OpenCV Display / Output]
    E --> I[EleGuard Web Dashboard\napp.py / src/dashboard.py]
```

---

## ✨ Key Features

- **🛡️ Secure Authentication**: Officer ID / Username & Password login portal with session management.
- **🚀 RT-DETR Vision Engine**: End-to-end transformer-based object detection delivering state-of-the-art accuracy and low latency.
- **🚨 Toggleable Multi-Channel Alert Controls**: Audio sirens (ON/OFF) and visual warning banners (ON/OFF) with customizable cooldown.
- **📊 Instant Status Indicators**: High-visibility **"🚨 YES — ELEPHANT DETECTED"** or **"🟢 NO — AREA CLEAR"** status badges and live count metrics.
- **📹 Multi-Source Surveillance**:
  - Image Upload & Built-in Sample Testing
  - Video File Surveillance & Tracking
  - Live Webcam Feed & RTSP Stream Integration
- **📁 Comprehensive Telemetry & Logging**: Real-time FPS / latency HUD and structured CSV audit logging (`logs/detections.csv`).

---

## 🛠️ Installation & Setup

```bash
# Clone repository
git clone https://github.com/satyabrata695/elephant-detection.git
cd elephant-detection

# Activate virtual environment
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Launching EleGuard Dashboard

Launch the EleGuard Web Application:

```bash
streamlit run app.py
```
*(Or `streamlit run src/dashboard.py`)*

### 🔑 Demo Login Credentials:
| Role | Username | Password |
| :--- | :--- | :--- |
| **Administrator** | `admin` | `eleguard2026` |
| **Forest Officer** | `forest_officer` | `ranger123` |
| **Guest / Demo** | `demo` | `demo123` |

### 3. Real-Time CLI Inference

**Live Webcam:**
```bash
python src/main.py --source 0
```

**Single Image Testing:**
```bash
python src/main.py --source "data/images/elephant1.png" --save
```

**Video File Processing:**
```bash
python src/main.py --source "data/videos/wildlife.mp4" --confidence 0.45 --save
```

**Headless RTSP IP Camera Stream (Edge / Server Deployment):**
```bash
python src/main.py --source "rtsp://username:password@192.168.1.100:554/stream" --no-display
```

---

## 🏋️ Model Training

To fine-tune RT-DETR on your custom dataset:

```bash
python train.py --epochs 25 --batch 4 --imgsz 640 --device cuda
```

### Training Options:
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--model` | `weights/rtdetr-l.pt` | Base model weights |
| `--data` | `data/data.yaml` | Dataset YAML configuration |
| `--epochs` | `25` | Number of training epochs |
| `--batch` | `4` | Training batch size |
| `--imgsz` | `640` | Training image resolution |
| `--device` | `auto` | Execution device (`cuda`, `cpu`) |
| `--workers` | `2` | DataLoader worker threads |
| `--resume` | `False` | Resume from last training checkpoint |

When training finishes, the best checkpoint is automatically saved to `weights/best.pt`.

---

## 📁 Repository Structure

```
elephant-detection/
├── configs/
│   └── config.py               # Global system configuration
├── data/
│   ├── data.yaml               # Roboflow dataset descriptor
│   └── images/
│       └── elephant1.png       # Sample test asset
├── logs/
│   └── .gitkeep                # Application logs & CSV audits
├── models/
│   └── .gitkeep                # Exported model weights
├── outputs/
│   └── .gitkeep                # Generated detection images & videos
├── src/
│   ├── core/
│   │   ├── detection.py        # Detection dataclass
│   │   └── frame_info.py       # FrameInfo telemetry dataclass
│   ├── alerts.py               # Audio & cooldown alert manager
│   ├── camera.py               # Unified video/webcam/RTSP stream source
│   ├── dashboard.py            # Streamlit web application
│   ├── detector.py             # RT-DETR inference engine
│   ├── logger.py               # Rotating & CSV audit logger
│   ├── main.py                 # CLI application orchestrator
│   ├── postprocessor.py        # IoU duplicate suppression & filtering
│   ├── preprocessor.py         # Image preprocessing
│   ├── utils.py                # Telemetry, FPS, & system diagnostics
│   └── visualize.py            # HUD overlay & bounding box renderer
├── .gitignore                  # Git size limit & artifact protection
├── requirements.txt            # Python dependencies
├── test.py                     # Diagnostic & benchmark test runner
├── train.py                    # Production model training controller
└── README.md                   # Project documentation
```

---

## ⚙️ Git & GitHub Push Information

This repository is optimized to adhere to GitHub's **100 MB per-file size limit** and **1 GB recommended repository size**:
- Checkpoints in `runs/` and `weights/*.pt` along with raw training datasets in `data/train/` are excluded by `.gitignore`.
- When cloning the repository, RT-DETR base weights (`rtdetr-l.pt`) are downloaded automatically by the system on first run.

---

## 📄 License
This project is licensed under the MIT License.
