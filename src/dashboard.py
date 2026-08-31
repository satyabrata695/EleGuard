"""
RT-DETR Elephant Detection System - Web Dashboard
Interactive Streamlit Dashboard providing Image/Video Inference, Live Camera Stream, and Detection Analytics.

Run command:
    streamlit run src/dashboard.py
"""

from __future__ import annotations
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.alerts import Alert, AlertManager
from src.core.frame_info import FrameInfo
from src.detector import ElephantDetector
from src.utils import get_system_info
from src.visualize import Visualizer


# =========================================================
# Streamlit Dashboard Class & Web App
# =========================================================

class Dashboard:
    """Streamlit Dashboard interface and controller."""

    def __init__(
        self,
        title: str = "🐘 RT-DETR Elephant Detection System",
        page_icon: str = "🐘",
        layout: str = "wide",
    ) -> None:
        self.title = title
        self.page_icon = page_icon
        self.layout = layout

    def render_app(self) -> None:
        """Render the complete Streamlit web dashboard."""
        st.set_page_config(
            page_title=self.title,
            page_icon=self.page_icon,
            layout=self.layout,
            initial_sidebar_state="expanded",
        )

        st.markdown(
            """
            <style>
            .main-header { font-size: 2.2rem; font-weight: 700; color: #1E88E5; margin-bottom: 0.2rem; }
            .sub-header { font-size: 1.05rem; color: #666; margin-bottom: 1.5rem; }
            .metric-card { background-color: #f8f9fa; border-radius: 8px; padding: 12px; border-left: 4px solid #1E88E5; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="main-header">🐘 RT-DETR Elephant Detection System</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">AI-Powered Early Warning & Wildlife Monitoring System | NIT Rourkela 5G MEC Lab</div>', unsafe_allow_html=True)

        # --------------------------------------------------
        # Sidebar Configuration
        # --------------------------------------------------
        st.sidebar.image("https://img.icons8.com/color/96/000000/elephant.png", width=70)
        st.sidebar.title("Model & Controls")

        root_dir = Path(__file__).resolve().parent.parent
        weights_dir = root_dir / "weights"
        available_weights = [str(p.relative_to(root_dir)) for p in weights_dir.glob("*.pt")] if weights_dir.exists() else []
        if not available_weights:
            available_weights = ["weights/rtdetr-l.pt", "rtdetr-l.pt"]

        selected_model = st.sidebar.selectbox("Model Weights", available_weights, index=0)
        model_path = root_dir / selected_model if not Path(selected_model).is_absolute() else Path(selected_model)

        conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.10, max_value=0.95, value=0.45, step=0.05)
        iou_threshold = st.sidebar.slider("IoU Duplicate Threshold", min_value=0.20, max_value=0.90, value=0.45, step=0.05)
        device_choice = st.sidebar.selectbox("Inference Device", ["auto", "cuda", "cpu"], index=0)

        st.sidebar.divider()
        sys_info = get_system_info()
        st.sidebar.caption(f"**Device:** {sys_info['gpu']}")
        st.sidebar.caption(f"**PyTorch:** {sys_info['pytorch']} | **CUDA:** {sys_info['cuda_available']}")

        # --------------------------------------------------
        # Initialize Cached Detector
        # --------------------------------------------------
        @st.cache_resource(show_spinner="Loading RT-DETR Model...")
        def load_detector(model_p: str, conf: float, iou: float, dev: str) -> ElephantDetector:
            return ElephantDetector(
                model_path=model_p,
                confidence=conf,
                iou=iou,
                device=dev,
                warmup=True,
            )

        try:
            detector = load_detector(str(model_path), conf_threshold, iou_threshold, device_choice)
            detector.set_confidence(conf_threshold)
            visualizer = Visualizer(show_fps=True, show_confidence=True, show_alert_banner=True)
            alert_manager = AlertManager(confidence_threshold=conf_threshold, sound_alert=False)
        except Exception as exc:
            st.error(f"Failed to load RT-DETR model from '{selected_model}': {exc}")
            return

        # --------------------------------------------------
        # Main Tabs
        # --------------------------------------------------
        tab_image, tab_video, tab_live, tab_analytics, tab_info = st.tabs([
            "📷 Image Detection",
            "🎥 Video File Inference",
            "📹 Live Stream / Webcam",
            "📊 Analytics & Logs",
            "ℹ️ System Info",
        ])

        # --------------------------------------------------
        # TAB 1: Image Detection
        # --------------------------------------------------
        with tab_image:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Upload or Select Image")
                sample_img_path = root_dir / "data" / "images" / "elephant1.png"
                use_sample = st.checkbox("Use sample elephant image", value=sample_img_path.exists())

                uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "bmp", "webp"])
                input_frame = None

                if uploaded_image is not None:
                    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                    input_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                elif use_sample and sample_img_path.exists():
                    input_frame = cv2.imread(str(sample_img_path))

                if input_frame is not None:
                    st.image(cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB), caption="Input Image", use_container_width=True)

            with col2:
                st.subheader("Detection Results")
                if input_frame is not None:
                    with st.spinner("Running RT-DETR inference..."):
                        frame_info = detector.detect(input_frame)
                        annotated_frame = visualizer.render(frame_info)
                        alerts = alert_manager.process(frame_info)

                    # Display metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Elephants Detected", frame_info.detection_count)
                    m2.metric("Latency", f"{frame_info.inference_time:.1f} ms")
                    m3.metric("Avg Confidence", f"{frame_info.average_confidence:.1%}")

                    if frame_info.has_detection:
                        st.error(f"🚨 **ALERT:** {frame_info.detection_count} elephant(s) detected with high confidence!")
                    else:
                        st.success("✓ No elephants detected.")

                    st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="Annotated Result", use_container_width=True)

                    if frame_info.detections:
                        det_df = pd.DataFrame([
                            {
                                "Class": d.class_name,
                                "Confidence": f"{d.confidence:.1%}",
                                "Box (x1, y1, x2, y2)": str(d.bbox),
                                "Area (px)": d.area,
                            }
                            for d in frame_info.detections
                        ])
                        st.dataframe(det_df, hide_index=True, use_container_width=True)

                        # Download button
                        _, buffer = cv2.imencode(".jpg", annotated_frame)
                        st.download_button(
                            label="⬇️ Download Annotated Image",
                            data=buffer.tobytes(),
                            file_name="elephant_detection_output.jpg",
                            mime="image/jpeg",
                        )

        # --------------------------------------------------
        # TAB 2: Video File Inference
        # --------------------------------------------------
        with tab_video:
            st.subheader("Video File Detection")
            uploaded_video = st.file_uploader("Upload video file", type=["mp4", "avi", "mov", "mkv"])
            if uploaded_video is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_video.read())
                tfile.close()

                cap = cv2.VideoCapture(tfile.name)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

                st.info(f"Video loaded: {total_frames} frames @ {vid_fps:.1f} FPS")
                start_btn = st.button("▶️ Process Video")
                video_placeholder = st.empty()
                progress_bar = st.progress(0)

                if start_btn:
                    frame_idx = 0
                    detected_count_total = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            break
                        frame_info = detector.detect(frame, frame_id=frame_idx, fps=vid_fps)
                        annotated = visualizer.render(frame_info)
                        if frame_info.has_detection:
                            detected_count_total += frame_info.detection_count

                        video_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                        frame_idx += 1
                        if total_frames > 0:
                            progress_bar.progress(min(1.0, frame_idx / total_frames))
                    cap.release()
                    st.success(f"Finished processing {frame_idx} frames. Total detections: {detected_count_total}")

        # --------------------------------------------------
        # TAB 3: Live Stream / Webcam
        # --------------------------------------------------
        with tab_live:
            st.subheader("Live Camera / RTSP Stream")
            source_input = st.text_input("Source (0 for webcam or RTSP/HTTP URL)", value="0")
            start_live = st.checkbox("Start Live Stream")
            live_placeholder = st.empty()
            metric_placeholder = st.empty()

            if start_live:
                src_val = int(source_input) if source_input.isdigit() else source_input
                cap_live = cv2.VideoCapture(src_val)
                if not cap_live.isOpened():
                    st.error(f"Cannot open video source: {source_input}")
                else:
                    fps_history = []
                    frame_num = 0
                    while start_live and cap_live.isOpened():
                        ret, frame = cap_live.read()
                        if not ret:
                            st.warning("Stream ended or disconnected.")
                            break
                        t0 = time.perf_counter()
                        frame_info = detector.detect(frame, frame_id=frame_num)
                        t1 = time.perf_counter()
                        fps_calc = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 30.0
                        frame_info.fps = fps_calc

                        annotated = visualizer.render(frame_info)
                        live_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

                        with metric_placeholder.container():
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Status", "ELEPHANT DETECTED" if frame_info.has_detection else "CLEAR")
                            c2.metric("FPS", f"{fps_calc:.1f}")
                            c3.metric("Latency", f"{frame_info.inference_time:.1f} ms")
                        frame_num += 1
                    cap_live.release()

        # --------------------------------------------------
        # TAB 4: Analytics & Audit Logs
        # --------------------------------------------------
        with tab_analytics:
            st.subheader("Detection Event Logs & Analytics")
            csv_path = root_dir / "logs" / "detections.csv"

            if csv_path.exists() and csv_path.stat().st_size > 0:
                try:
                    df = pd.read_csv(csv_path)
                    if not df.empty:
                        st.dataframe(df.tail(100), use_container_width=True)
                        st.download_button(
                            "⬇️ Export Full Audit CSV",
                            data=df.to_csv(index=False).encode("utf-8"),
                            file_name="detections_audit.csv",
                            mime="text/csv",
                        )
                    else:
                        st.info("No detections recorded yet in logs/detections.csv.")
                except Exception as exc:
                    st.error(f"Error reading CSV logs: {exc}")
            else:
                st.info("No detection log file found. Detection events will appear here once main.py runs.")

        # --------------------------------------------------
        # TAB 5: System & Model Information
        # --------------------------------------------------
        with tab_info:
            st.subheader("Hardware & Software Environment")
            st.json(sys_info)

            st.subheader("Detector Metadata")
            st.json(detector.statistics())


# Standalone entry point
if __name__ == "__main__":
    app = Dashboard()
    app.render_app()