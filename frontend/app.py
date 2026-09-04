"""
EleGuard - AI-Powered Elephant Detection & Early Warning System
Streamlit Web Application Frontend

Features:
- Secure Authentication & Session Management (top-right account controls)
- Multi-Source Inference (Image Upload, Video Upload, Live Camera / Webcam)
- Audio Alert and Visual Banner Toggle Controls (ON / OFF)
- Clear, alert-toned "Elephant Detected" status & total count badges
- Detailed Bounding Box Telemetry & Exportable Audit Reports
- Session Detection Analytics & Snooze-able Alerting
"""

from __future__ import annotations
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.alerts import Alert, AlertManager
from backend.core.frame_info import FrameInfo
from backend.detector import ElephantDetector
from backend.utils import get_system_info
from backend.visualize import Visualizer
from frontend.components.audio import play_browser_alarm_script
from frontend.components.styles import get_custom_css


# --------------------------------------------------------------------------- #
# Extra CSS: alert-toned status cards, top-right account bar, collapsible
# sidebar sections. Appended on top of get_custom_css() so the existing
# stylesheet keeps working untouched.
# --------------------------------------------------------------------------- #
def get_extra_css() -> str:
    """Extra CSS injected on top of the design system for login-page and misc overrides."""
    return """
    <style>
    /* Login page centered card */
    .login-card {
        background: rgba(15,23,42,0.85);
        border: 1px solid rgba(16,185,129,0.18);
        border-radius: 22px;
        padding: 36px 32px 28px;
        backdrop-filter: blur(24px);
        box-shadow: 0 30px 60px -15px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.05);
        position: relative;
        overflow: hidden;
    }
    .login-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent 0%, #10B981 50%, transparent 100%);
    }
    .login-heading {
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.03em;
        margin-bottom: 4px;
    }
    .login-subheading {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.82rem;
        color: #475569;
        margin-bottom: 20px;
        letter-spacing: 0.01em;
    }

    /* Credential hint card */
    .cred-hint {
        background: rgba(16,185,129,0.05);
        border: 1px solid rgba(16,185,129,0.15);
        border-radius: 12px;
        padding: 14px 16px;
    }
    .cred-hint p { font-family: 'Space Grotesk', sans-serif; font-size: 0.82rem; color: #94A3B8; margin: 0 0 6px 0; }
    .cred-row {
        display: flex; align-items: center; gap: 10px;
        padding: 6px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .cred-row:last-child { border-bottom: none; }
    .cred-badge {
        font-family: 'Space Mono', monospace;
        font-size: 0.68rem;
        background: rgba(16,185,129,0.15);
        color: #10B981;
        padding: 2px 8px;
        border-radius: 6px;
        border: 1px solid rgba(16,185,129,0.2);
        white-space: nowrap;
    }
    .cred-text {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.8rem;
        color: #64748B;
    }
    </style>
    """


class EleGuardApp:
    """EleGuard Web Application & Surveillance Dashboard."""

    CREDENTIALS = {
        "admin": "eleguard2026",
        "forest_officer": "ranger123",
        "demo": "demo123",
    }

    ROLE_LABELS = {
        "admin": "System Administrator",
        "forest_officer": "Forest Ranger / Officer",
        "demo": "Guest Observer",
    }

    def __init__(self) -> None:
        self.title = "EleGuard | AI Elephant Detection & Early Warning System"
        self._st = None

    def _get_st(self):
        """Import Streamlit lazily."""
        if self._st is None:
            try:
                import streamlit as st
                self._st = st
            except ImportError as exc:
                raise ImportError(
                    "Streamlit is required to launch EleGuard Dashboard.\n"
                    "Install it using: pip install streamlit\n"
                    "Then run: streamlit run app.py"
                ) from exc
        return self._st

    # ------------------------------------------------------------------ #
    # LOGIN SCREEN
    # ------------------------------------------------------------------ #
    def render_login(self) -> None:
        """Render the secure login screen."""
        st = self._get_st()
        st.markdown(get_custom_css(), unsafe_allow_html=True)
        st.markdown(get_extra_css(), unsafe_allow_html=True)

        # Centered brand logo at top
        st.markdown(
            """
            <div style="text-align:center; padding: 40px 0 28px 0;">
                <div style="font-size: 4rem; filter: drop-shadow(0 0 20px rgba(16,185,129,0.7));">
                    🐘
                </div>
                <h1 style="font-family:'Syne',sans-serif; font-size:2.6rem; font-weight:800;
                           letter-spacing:-0.05em; color:#F8FAFC; margin:8px 0 0 0;">
                    Ele<span style="background:linear-gradient(135deg,#10B981,#34D399);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text;">Guard</span>
                </h1>
                <p style="font-family:'Space Grotesk',sans-serif; color:#475569;
                          font-size:0.9rem; margin:6px 0 0 0; letter-spacing:0.02em;">
                    AI Wildlife Perimeter Surveillance &amp; Early Warning Portal
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns([1, 1.8, 1])
        with col2:
            st.markdown(
                '<div class="login-card">'
                '<div class="login-heading">🔒 Secure Access</div>'
                '<div class="login-subheading">Authenticate with authorized credentials to enter the surveillance portal.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            # Note: form rendered outside the HTML card div due to Streamlit constraints
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username / Officer ID", placeholder="e.g. admin", label_visibility="visible")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
                submit_btn = st.form_submit_button(
                    "🚀  Sign In to EleGuard",
                    use_container_width=True,
                    type="primary",
                )

                if submit_btn:
                    user_clean = username.strip().lower()
                    if user_clean in self.CREDENTIALS and self.CREDENTIALS[user_clean] == password:
                        st.session_state["authenticated"] = True
                        st.session_state["user"] = user_clean
                        st.session_state["login_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.setdefault("session_detection_count", 0)
                        st.session_state.setdefault("alerts_snoozed", False)
                        st.success(f"✓ Access granted. Welcome, {user_clean.replace('_', ' ').title()}!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Please check your username and password.")

            st.markdown(
                """
                <div class="cred-hint" style="margin-top:18px;">
                    <p>🔑 Demo Credentials</p>
                    <div class="cred-row">
                        <span class="cred-badge">ADMIN</span>
                        <span class="cred-text">admin &nbsp;/&nbsp; eleguard2026</span>
                    </div>
                    <div class="cred-row">
                        <span class="cred-badge">RANGER</span>
                        <span class="cred-text">forest_officer &nbsp;/&nbsp; ranger123</span>
                    </div>
                    <div class="cred-row">
                        <span class="cred-badge">GUEST</span>
                        <span class="cred-text">demo &nbsp;/&nbsp; demo123</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------------------ #
    # TOP-RIGHT ACCOUNT BAR
    # ------------------------------------------------------------------ #
    def render_topbar(self) -> None:
        """Render the top bar with LIVE badge on the left and user + logout on the right."""
        st = self._get_st()
        user = st.session_state.get("user", "officer")
        role = self.ROLE_LABELS.get(user, "Field User")
        initials = "".join([p[0] for p in user.split("_")]).upper()[:2]
        login_time = st.session_state.get("login_time", "")

        # Layout: live badge | spacer | user pill + logout button
        col_left, col_mid, col_user, col_logout = st.columns([2, 3, 2, 1])

        with col_left:
            st.markdown(
                """
                <div style="padding-top:6px;">
                    <span class="live-badge">
                        <span class="live-dot"></span>
                        LIVE SURVEILLANCE
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_mid:
            if login_time:
                st.markdown(
                    f'<div style="font-family:\'Space Mono\',monospace; font-size:0.65rem;'
                    f' color:#334155; padding-top:10px; text-align:center;">'
                    f'Session: {login_time}</div>',
                    unsafe_allow_html=True,
                )

        with col_user:
            st.markdown(
                f"""
                <div class="topbar-user" style="justify-content:flex-end;">
                    <div class="avatar">{initials}</div>
                    <div>
                        <div class="name">{user.replace('_', ' ').title()}</div>
                        <div class="role">{role}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_logout:
            st.markdown('<div style="padding-top:4px;">', unsafe_allow_html=True)
            if st.button("⏻ Logout", use_container_width=True, key="topbar_logout",
                         help="Sign out and return to the login screen"):
                st.session_state["authenticated"] = False
                st.session_state["user"] = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------------------------------------------------ #
    # DASHBOARD
    # ------------------------------------------------------------------ #
    def render_dashboard(self) -> None:
        """Render the full EleGuard Surveillance Dashboard once authenticated."""
        st = self._get_st()
        st.markdown(get_custom_css(), unsafe_allow_html=True)
        st.markdown(get_extra_css(), unsafe_allow_html=True)

        root_dir = PROJECT_ROOT
        weights_dir = root_dir / "weights"

        st.session_state.setdefault("session_detection_count", 0)
        st.session_state.setdefault("alerts_snoozed", False)

        # ---- Top-right account bar (replaces sidebar login/logout) ----
        self.render_topbar()

        # ---- Brand Header ----
        session_count = st.session_state.get("session_detection_count", 0)
        st.markdown(
            f"""
            <div class="brand-container">
                <div class="brand-logo">🐘</div>
                <div style="flex-grow: 1;">
                    <h1 class="brand-title">Ele<span>Guard</span></h1>
                    <p class="brand-subtitle">Real-Time Wildlife Early Warning &amp; Automated Conflict Mitigation System</p>
                </div>
                <div style="text-align:right; flex-shrink:0;">
                    <div style="font-family:'Space Mono',monospace; font-size:0.62rem; color:#334155;
                                text-transform:uppercase; letter-spacing:0.1em; margin-bottom:4px;">Session Alerts</div>
                    <div style="font-family:'Syne',sans-serif; font-size:2rem; font-weight:800;
                                color:{'#FCA5A5' if session_count > 0 else '#10B981'};
                                line-height:1; letter-spacing:-0.04em;">{session_count}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ------------------------------------------------------------ #
        # HORIZONTAL CONTROL BAR (replaces sidebar)
        # ------------------------------------------------------------ #
        # ---- Row 1: always-visible quick controls ----
        st.markdown('<div class="ctrl-bar">', unsafe_allow_html=True)
        cb1, cb2, cb3, cb4, cb5, cb6, cb7 = st.columns([1.6, 1, 1, 1.4, 1.4, 1.4, 1])

        with cb1:
            available_weights = (
                [str(p.relative_to(root_dir)) for p in weights_dir.glob("*.pt")]
                if weights_dir.exists()
                else []
            )
            if available_weights:
                available_weights.sort(key=lambda x: (0 if "best" in Path(x).name else 1, x))
            if not available_weights:
                available_weights = ["weights/best.pt", "weights/rtdetr-l.pt"]
            selected_model = st.selectbox(
                "Model", available_weights, index=0,
                label_visibility="visible"
            )
            model_path = root_dir / selected_model if not Path(selected_model).is_absolute() else Path(selected_model)

        with cb2:
            device_choice = st.selectbox("Hardware", ["auto", "cuda", "cpu"], index=0)

        with cb3:
            conf_threshold = st.slider(
                "Confidence", min_value=0.10, max_value=0.95, value=0.45, step=0.05,
                help="Filter detections below this confidence score."
            )

        with cb4:
            iou_threshold = st.slider(
                "IoU / NMS", min_value=0.20, max_value=0.90, value=0.45, step=0.05
            )

        with cb5:
            alert_cooldown = st.slider(
                "Cooldown (s)", min_value=1.0, max_value=60.0, value=8.0, step=1.0,
                help="Minimum seconds between audio alarms."
            )

        with cb6:
            enable_sound = st.toggle(
                "🔊 Audio Alarm", value=True,
                help="Emit audio siren on detection."
            )
            enable_banner = st.toggle(
                "🚨 Visual Banner", value=True,
                help="Show warning banner overlay on frame."
            )

        with cb7:
            snooze = st.toggle(
                "🔕 Snooze", value=st.session_state["alerts_snoozed"],
                help="Mute all alerts temporarily."
            )
            st.session_state["alerts_snoozed"] = snooze
            if snooze:
                enable_sound = False
            show_fps = st.toggle("📹 FPS", value=True)
            show_conf_overlay = st.toggle("🎯 Conf", value=True)
            compact_tables = st.toggle("📊 Compact", value=False)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)

        # Cached Detector Initialization
        @st.cache_resource(show_spinner="Loading EleGuard RT-DETR Engine...")
        def get_detector(m_path: str, conf: float, iou: float, dev: str) -> ElephantDetector:
            return ElephantDetector(
                model_path=m_path,
                confidence=conf,
                iou=iou,
                device=dev,
                target_classes=("elephant",),
                warmup=True,
            )

        try:
            detector = get_detector(str(model_path), conf_threshold, iou_threshold, device_choice)
            detector.set_confidence(conf_threshold)
            visualizer = Visualizer(
                show_fps=show_fps,
                show_confidence=show_conf_overlay,
                show_alert_banner=enable_banner,
            )
            alert_manager = AlertManager(
                confidence_threshold=conf_threshold,
                cooldown_seconds=alert_cooldown,
                sound_alert=enable_sound,
            )
        except Exception as exc:
            st.error(f"Failed to initialize EleGuard AI Engine from '{selected_model}': {exc}")
            return

        # Navigation Tabs
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        tab_image, tab_video, tab_camera, tab_logs, tab_analytics, tab_info = st.tabs([
            "📷  Image Analysis",
            "🎥  Video Surveillance",
            "📹  Live Camera Feed",
            "📋  Audit Logs",
            "📈  Analytics",
            "⚙️  System Status",
        ])

        # ============================================================ #
        # TAB 1: Image Detection
        # ============================================================ #
        with tab_image:
            col_in, col_out = st.columns([1, 1], gap="medium")

            with col_in:
                st.markdown("#### 1. Input Image Source")
                sample_path = root_dir / "data" / "images" / "elephant1.png"
                use_sample = st.checkbox("📁 Load built-in sample elephant image", value=False)

                uploaded_img = st.file_uploader(
                    "Upload Image File",
                    type=["jpg", "jpeg", "png", "webp", "bmp"],
                    help="Upload aerial, railway track, or forest camera snapshot",
                )

                input_img = None
                img_source_name = ""
                if uploaded_img is not None:
                    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
                    input_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    img_source_name = uploaded_img.name
                elif use_sample and sample_path.exists():
                    input_img = cv2.imread(str(sample_path))
                    img_source_name = sample_path.name

                if input_img is not None:
                    h, w, c = input_img.shape
                    st.image(
                        cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB),
                        caption=f"Selected Input: {img_source_name} ({w}x{h} px)",
                        use_container_width=True,
                    )

                    c_btn1, c_btn2 = st.columns([2, 1])
                    with c_btn1:
                        analyze_clicked = st.button("🔍 Run Elephant Detection", type="primary", use_container_width=True)
                    with c_btn2:
                        auto_analyze = st.checkbox("⚡ Auto-run", value=True, help="Automatically run detection when new image is chosen")

                    current_img_key = f"{img_source_name}_{h}_{w}"
                    if "last_image_key" not in st.session_state or st.session_state["last_image_key"] != current_img_key:
                        st.session_state["last_image_key"] = current_img_key
                        st.session_state["image_analyzed"] = False
                        st.session_state["image_result"] = None

                    should_run = analyze_clicked or (auto_analyze and not st.session_state.get("image_analyzed", False))

                    if should_run:
                        with st.spinner("🤖 Analyzing frame with RT-DETR..."):
                            try:
                                t0 = time.perf_counter()
                                f_info = detector.detect(input_img)
                                annotated_bgr = visualizer.render(f_info)
                                if enable_sound and not st.session_state["alerts_snoozed"]:
                                    alert_manager.process(f_info)
                                latency_ms = (time.perf_counter() - t0) * 1000.0

                                if f_info.has_detection:
                                    st.session_state["session_detection_count"] += 1

                                st.session_state["image_analyzed"] = True
                                st.session_state["image_result"] = {
                                    "frame_info": f_info,
                                    "annotated_bgr": annotated_bgr,
                                    "latency_ms": latency_ms,
                                }
                            except Exception as exc:
                                st.error(f"❌ Detection failed: {exc}")
                else:
                    st.info("👆 Please upload an image or check the sample image above to start inference.")

            with col_out:
                st.markdown("#### 2. Detection Results & Status")

                cached_res = st.session_state.get("image_result", None)
                if input_img is not None and cached_res is not None:
                    frame_info = cached_res["frame_info"]
                    annotated_bgr = cached_res["annotated_bgr"]
                    has_elephant = frame_info.has_detection
                    elephant_count = frame_info.detection_count

                    if has_elephant:
                        st.markdown(
                            """
                            <div class="status-card-alert">
                                <div class="status-title">Perimeter Status</div>
                                <div class="status-value-alert">⚠️ ELEPHANT DETECTED — TAKE ACTION</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if enable_sound and not st.session_state["alerts_snoozed"]:
                            st.components.v1.html(play_browser_alarm_script(), height=0)
                    else:
                        st.markdown(
                            """
                            <div class="status-card-clear">
                                <div class="status-title">Perimeter Status</div>
                                <div class="status-value-clear">🟢 AREA CLEAR — NO ELEPHANT DETECTED</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(
                            f"""
                            <div class="metric-pill">
                                <div class="metric-pill-label">Elephant Count</div>
                                <div class="metric-pill-val" style="color: {'#EF4444' if has_elephant else '#10B981'};">
                                    {elephant_count}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with m2:
                        st.markdown(
                            f"""
                            <div class="metric-pill">
                                <div class="metric-pill-label">Inference Latency</div>
                                <div class="metric-pill-val">{frame_info.inference_time:.1f} ms</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with m3:
                        avg_conf = f"{frame_info.average_confidence:.1%}" if has_elephant else "N/A"
                        st.markdown(
                            f"""
                            <div class="metric-pill">
                                <div class="metric-pill-label">Avg Confidence</div>
                                <div class="metric-pill-val">{avg_conf}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                    st.image(
                        cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
                        caption="EleGuard Annotated Visual Result",
                        use_container_width=True,
                    )

                    if frame_info.detections:
                        st.markdown(
                            """
                            <div style="display:flex; align-items:center; gap:10px; margin:16px 0 10px 0;">
                                <div style="width:28px;height:28px; background:rgba(255,255,255,0.05);
                                           border:1px solid rgba(255,255,255,0.1); border-radius:8px;
                                           display:flex;align-items:center;justify-content:center;font-size:0.85rem;">📋</div>
                                <span style="font-family:'Syne',sans-serif; font-size:0.95rem; font-weight:700;
                                            color:#CBD5E1; letter-spacing:-0.01em;">Detection Telemetry</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        det_data = [
                            {
                                "Object": f"#{i+1}",
                                "Class": d.class_name.capitalize(),
                                "Confidence": f"{d.confidence:.1%}",
                                "Bounding Box [x1, y1, x2, y2]": f"[{d.bbox[0]}, {d.bbox[1]}, {d.bbox[2]}, {d.bbox[3]}]",
                                "Area": f"{d.area:,} px²",
                            }
                            for i, d in enumerate(frame_info.detections)
                        ]
                        df_det = pd.DataFrame(det_data)
                        if compact_tables:
                            st.dataframe(df_det, hide_index=True, use_container_width=True, height=180)
                        else:
                            st.dataframe(df_det, hide_index=True, use_container_width=True)

                        c_dl1, c_dl2 = st.columns(2)
                        with c_dl1:
                            _, encoded = cv2.imencode(".jpg", annotated_bgr)
                            st.download_button(
                                label="⬇️ Download Annotated Image",
                                data=encoded.tobytes(),
                                file_name=f"eleguard_detection_{int(time.time())}.jpg",
                                mime="image/jpeg",
                                use_container_width=True,
                            )
                        with c_dl2:
                            st.download_button(
                                label="⬇️ Export Telemetry CSV",
                                data=df_det.to_csv(index=False).encode("utf-8"),
                                file_name=f"detection_telemetry_{int(time.time())}.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )
                    else:
                        st.info("No objects detected in the current frame above the confidence threshold.")
                elif input_img is not None:
                    st.info("📸 Image loaded! Click **'🔍 Run Elephant Detection'** on the left to analyze.")
                else:
                    st.markdown(
                        """
                        <div class="empty-state">
                            <div class="empty-state-icon">📷</div>
                            <h4>Awaiting Image Input</h4>
                            <p>Upload a photo or enable the sample image on the left,<br>then click <b>Run Elephant Detection</b> to begin analysis.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # ============================================================ #
        # TAB 2: Video File Inference
        # ============================================================ #
        with tab_video:
            st.markdown("#### 🎥 Video File Surveillance")
            uploaded_vid = st.file_uploader(
                "Upload Video File for Automated Tracking",
                type=["mp4", "avi", "mov", "mkv"],
            )

            if uploaded_vid is not None:
                temp_vid = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_vid.write(uploaded_vid.read())
                temp_vid.close()

                cap = cv2.VideoCapture(temp_vid.name)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                v_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

                st.info(f"Video Loaded: **{total_frames} frames** @ **{v_fps:.1f} FPS**")

                col_btn, col_skip = st.columns([1, 1])
                with col_btn:
                    process_btn = st.button("▶️ Start Video Detection Pipeline", use_container_width=True)
                with col_skip:
                    frame_skip = st.slider("Frame Skip Rate (Speed vs Precision)", 1, 5, 1)

                video_screen = st.empty()
                status_box = st.empty()
                progress = st.progress(0)

                if process_btn:
                    curr_frame = 0
                    total_elephant_detections = 0

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            break

                        if curr_frame % frame_skip == 0:
                            f_info = detector.detect(frame, frame_id=curr_frame, fps=v_fps)
                            annotated = visualizer.render(f_info)

                            if f_info.has_detection:
                                total_elephant_detections += f_info.detection_count
                                st.session_state["session_detection_count"] += 1
                                if enable_sound and not st.session_state["alerts_snoozed"]:
                                    alert_manager.process(f_info)

                            video_screen.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

                            status_box.markdown(
                                f"""
                                <div style="display: flex; gap: 16px; justify-content: center; margin-bottom: 10px;">
                                    <span style="font-weight: 700; color: {'#EF4444' if f_info.has_detection else '#10B981'};">
                                        Status: {'⚠️ ELEPHANT DETECTED' if f_info.has_detection else '🟢 CLEAR'}
                                    </span>
                                    <span>| Frame: <b>{curr_frame}/{total_frames}</b></span>
                                    <span>| Latency: <b>{f_info.inference_time:.1f}ms</b></span>
                                    <span>| Current Count: <b>{f_info.detection_count}</b></span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        curr_frame += 1
                        if total_frames > 0:
                            progress.progress(min(1.0, curr_frame / total_frames))

                    cap.release()
                    st.success(f"✓ Video processing complete! Total elephant sightings across frames: {total_elephant_detections}")

        # ============================================================ #
        # TAB 3: Direct Camera Feed / Live Webcam
        # ============================================================ #
        with tab_camera:
            st.markdown("#### 📹 Direct Camera & Real-Time Webcam Stream")

            cam_mode = st.radio(
                "Select Camera Input Mode",
                ["📸 Instant Snapshot Camera", "🔴 Continuous Live Stream (OpenCV / RTSP)"],
                horizontal=True,
            )

            if cam_mode == "📸 Instant Snapshot Camera":
                camera_photo = st.camera_input("Take a photo with your device camera")
                if camera_photo is not None:
                    bytes_data = camera_photo.getvalue()
                    np_arr = np.frombuffer(bytes_data, np.uint8)
                    cam_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    with st.spinner("Analyzing camera snapshot..."):
                        f_info = detector.detect(cam_frame)
                        annotated = visualizer.render(f_info)

                    if f_info.has_detection:
                        st.session_state["session_detection_count"] += 1
                        st.markdown(
                            f"""
                            <div class="status-card-alert">
                                <div class="status-title">Camera Live Detection</div>
                                <div class="status-value-alert">⚠️ {f_info.detection_count} ELEPHANT(S) DETECTED</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if enable_sound and not st.session_state["alerts_snoozed"]:
                            st.components.v1.html(play_browser_alarm_script(), height=0)
                    else:
                        st.markdown(
                            """
                            <div class="status-card-clear">
                                <div class="status-title">Camera Live Detection</div>
                                <div class="status-value-clear">🟢 NO ELEPHANT DETECTED</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            else:
                source_val = st.text_input("Live Camera Index or RTSP URL", value="0")
                start_live = st.checkbox("🟢 Start Live Stream")

                stream_screen = st.empty()
                stream_metrics = st.empty()

                if start_live:
                    src_parsed = int(source_val) if source_val.isdigit() else source_val
                    cap_live = cv2.VideoCapture(src_parsed)

                    if not cap_live.isOpened():
                        st.error(f"Cannot connect to video stream source: {source_val}")
                    else:
                        frame_i = 0
                        while start_live and cap_live.isOpened():
                            ret, frame = cap_live.read()
                            if not ret:
                                st.warning("Camera stream disconnected or ended.")
                                break

                            t_start = time.perf_counter()
                            f_info = detector.detect(frame, frame_id=frame_i)
                            t_end = time.perf_counter()
                            fps_live = 1.0 / (t_end - t_start) if (t_end - t_start) > 0 else 30.0
                            f_info.fps = fps_live

                            if f_info.has_detection:
                                st.session_state["session_detection_count"] += 1

                            annotated = visualizer.render(f_info)
                            stream_screen.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

                            with stream_metrics.container():
                                c1, c2, c3 = st.columns(3)
                                with c1:
                                    st.markdown(
                                        f"""
                                        <div class="metric-pill">
                                            <div class="metric-pill-label">Live Status</div>
                                            <div class="metric-pill-val" style="color: {'#EF4444' if f_info.has_detection else '#10B981'};">
                                                {'⚠️ ALERT' if f_info.has_detection else '🟢 CLEAR'}
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                                with c2:
                                    st.markdown(
                                        f"""
                                        <div class="metric-pill">
                                            <div class="metric-pill-label">Detected Count</div>
                                            <div class="metric-pill-val">{f_info.detection_count}</div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                                with c3:
                                    st.markdown(
                                        f"""
                                        <div class="metric-pill">
                                            <div class="metric-pill-label">Real-Time FPS</div>
                                            <div class="metric-pill-val">{fps_live:.1f} FPS</div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                            frame_i += 1
                        cap_live.release()

        # ============================================================ #
        # TAB 4: Audit Logs & History
        # ============================================================ #
        with tab_logs:
            st.markdown(
                """
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:18px;">
                    <div style="width:38px;height:38px; background:linear-gradient(135deg,rgba(16,185,129,0.2),rgba(5,150,105,0.15));
                               border:1px solid rgba(16,185,129,0.3); border-radius:11px;
                               display:flex;align-items:center;justify-content:center;font-size:1.1rem;">📋</div>
                    <div>
                        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
                                    color:#E2E8F0;letter-spacing:-0.02em;">Detection Audit Logs</div>
                        <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#334155;
                                    text-transform:uppercase;letter-spacing:0.06em;">Event history &amp; exportable reports</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            csv_path = root_dir / "logs" / "detections.csv"

            if csv_path.exists() and csv_path.stat().st_size > 0:
                try:
                    df = pd.read_csv(csv_path)
                    if not df.empty:
                        st.dataframe(df.tail(100), use_container_width=True)
                        st.download_button(
                            "⬇️ Export EleGuard Detection Audit CSV",
                            data=df.to_csv(index=False).encode("utf-8"),
                            file_name=f"eleguard_audit_{int(time.time())}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    else:
                        st.info("No detections recorded yet in logs/detections.csv.")
                except Exception as exc:
                    st.error(f"Error reading CSV logs: {exc}")
            else:
                st.info("No detection log file found. Detection events will appear here once surveillance streams run.")

        # ============================================================ #
        # TAB 5: Analytics (new)
        # ============================================================ #
        with tab_analytics:
            st.markdown(
                """
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:18px;">
                    <div style="width:38px;height:38px; background:linear-gradient(135deg,rgba(139,92,246,0.2),rgba(109,40,217,0.15));
                               border:1px solid rgba(139,92,246,0.3); border-radius:11px;
                               display:flex;align-items:center;justify-content:center;font-size:1.1rem;">📈</div>
                    <div>
                        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
                                    color:#E2E8F0;letter-spacing:-0.02em;">Detection Analytics</div>
                        <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#334155;
                                    text-transform:uppercase;letter-spacing:0.06em;">Trends, distributions &amp; session stats</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            csv_path = root_dir / "logs" / "detections.csv"

            if csv_path.exists() and csv_path.stat().st_size > 0:
                try:
                    df = pd.read_csv(csv_path)
                    if not df.empty and "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                        df = df.dropna(subset=["timestamp"])
                        daily = df.set_index("timestamp").resample("D").size()
                        st.markdown("##### Detections Per Day")
                        st.line_chart(daily)

                        if "confidence" in df.columns:
                            st.markdown("##### Confidence Distribution")
                            st.bar_chart(df["confidence"].round(1).value_counts().sort_index())
                    else:
                        st.info("Not enough structured data yet to chart trends. Run more detections first.")
                except Exception as exc:
                    st.error(f"Could not build analytics from logs: {exc}")
            else:
                st.info("No detection log file found yet — analytics will populate once surveillance runs generate logs/detections.csv.")

            st.markdown("##### This Session")
            st.metric("Total Alerts This Session", st.session_state.get("session_detection_count", 0))

        # ============================================================ #
        # TAB 6: System & Hardware Status
        # ============================================================ #
        with tab_info:
            st.markdown(
                """
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:18px;">
                    <div style="width:38px;height:38px; background:linear-gradient(135deg,rgba(59,130,246,0.2),rgba(37,99,235,0.15));
                               border:1px solid rgba(59,130,246,0.3); border-radius:11px;
                               display:flex;align-items:center;justify-content:center;font-size:1.1rem;">⚙️</div>
                    <div>
                        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
                                    color:#E2E8F0;letter-spacing:-0.02em;">System Status</div>
                        <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#334155;
                                    text-transform:uppercase;letter-spacing:0.06em;">Hardware &amp; AI engine telemetry</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            sys_info = get_system_info()

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🖥️ Hardware & AI Accelerators")
                st.json(sys_info)

            with c2:
                st.subheader("🧠 Detector Telemetry")
                st.json(detector.statistics())

    def run(self) -> None:
        """Main application lifecycle controller."""
        st = self._get_st()
        st.set_page_config(
            page_title="EleGuard | AI Elephant Detection System",
            page_icon="🐘",
            layout="wide",
            initial_sidebar_state="collapsed",
        )

        if "authenticated" not in st.session_state:
            st.session_state["authenticated"] = False

        if not st.session_state["authenticated"]:
            self.render_login()
        else:
            self.render_dashboard()


if __name__ == "__main__":
    app = EleGuardApp()
    app.run()