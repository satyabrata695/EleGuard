"""
EleGuard Premium Design System & CSS Tokens
Ultra-premium dark glassmorphism theme with animated gradients,
Syne + Space Grotesk typography, and rich visual polish.
"""


def get_custom_css() -> str:
    """Return complete premium CSS stylesheet for EleGuard dashboard."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

    /* Hide the sidebar collapse arrow since sidebar is unused */
    [data-testid="collapsedControl"] { display: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }

    /* ================================================================
       GLOBAL RESET & BASE TYPOGRAPHY
    ================================================================ */
    html, body, [class*="css"], .stApp {
        font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #020817 !important;
        color: #E2E8F0 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Syne', sans-serif !important;
        letter-spacing: -0.02em !important;
    }

    /* ================================================================
       ANIMATED BACKGROUND MESH
    ================================================================ */
    .stApp {
        background:
            radial-gradient(ellipse 80% 50% at 20% -10%, rgba(16,185,129,0.07) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 80% 110%, rgba(59,130,246,0.06) 0%, transparent 60%),
            radial-gradient(ellipse 50% 30% at 50% 50%, rgba(139,92,246,0.04) 0%, transparent 70%),
            #020817 !important;
    }

    /* ================================================================
       SCROLLBAR
    ================================================================ */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #0F172A; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #10B981, #059669); border-radius: 99px; }

    /* ================================================================
       STREAMLIT WIDGET OVERRIDES
    ================================================================ */
    .stButton > button {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        letter-spacing: 0.01em !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3) !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(16,185,129,0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 8px 25px rgba(16,185,129,0.5) !important;
    }

    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stTextArea textarea {
        font-family: 'Space Grotesk', sans-serif !important;
        background: rgba(15,23,42,0.8) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: #E2E8F0 !important;
        transition: border-color 0.2s ease !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea textarea:focus {
        border-color: #10B981 !important;
        box-shadow: 0 0 0 2px rgba(16,185,129,0.15) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15,23,42,0.6) !important;
        border-radius: 14px !important;
        padding: 5px !important;
        gap: 4px !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        backdrop-filter: blur(10px) !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.02em !important;
        border-radius: 10px !important;
        color: #64748B !important;
        padding: 8px 16px !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(5,150,105,0.15)) !important;
        color: #10B981 !important;
        border: 1px solid rgba(16,185,129,0.3) !important;
    }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A1628 0%, #0D1F3C 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }
    [data-testid="stSidebar"] > div { padding: 20px 16px !important; }

    .stExpander {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 12px !important;
        margin-bottom: 10px !important;
    }
    .stExpander summary {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        color: #CBD5E1 !important;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #10B981, #34D399) !important;
        border-radius: 99px !important;
    }

    /* Info/Success/Error */
    .stAlert {
        border-radius: 12px !important;
        border-left-width: 3px !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }

    /* Dataframe */
    .stDataFrame {
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    /* Slider */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #10B981, #059669) !important;
    }

    /* Toggle & Radio */
    .stCheckbox label, .stRadio label {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.88rem !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(16,185,129,0.3) !important;
        border-radius: 14px !important;
        background: rgba(16,185,129,0.03) !important;
        transition: border-color 0.2s ease !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(16,185,129,0.6) !important;
        background: rgba(16,185,129,0.06) !important;
    }

    /* Hide default streamlit chrome */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* ================================================================
       BRAND HEADER
    ================================================================ */
    .brand-container {
        display: flex;
        align-items: center;
        gap: 20px;
        padding: 20px 28px;
        background: linear-gradient(135deg, rgba(15,23,42,0.95) 0%, rgba(20,30,55,0.95) 100%);
        border-radius: 18px;
        border: 1px solid rgba(16,185,129,0.15);
        margin-bottom: 28px;
        box-shadow: 0 20px 40px -10px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05);
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
    }
    .brand-container::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(16,185,129,0.5), transparent);
    }
    .brand-container::after {
        content: '';
        position: absolute;
        top: -50%; right: -10%;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(16,185,129,0.06) 0%, transparent 70%);
        pointer-events: none;
    }
    .brand-logo {
        font-size: 3rem;
        line-height: 1;
        filter: drop-shadow(0 0 16px rgba(16,185,129,0.6));
        animation: floatLogo 3s ease-in-out infinite;
    }
    @keyframes floatLogo {
        0%, 100% { transform: translateY(0); }
        50%       { transform: translateY(-4px); }
    }
    .brand-title {
        font-family: 'Syne', sans-serif !important;
        font-size: 2.1rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 0;
        letter-spacing: -0.04em;
        line-height: 1;
    }
    .brand-title span {
        background: linear-gradient(135deg, #10B981, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .brand-subtitle {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.88rem;
        color: #64748B;
        margin: 5px 0 0 0;
        letter-spacing: 0.01em;
    }

    /* ================================================================
       STATUS CARDS
    ================================================================ */
    .status-card-yes, .status-card-alert {
        background: linear-gradient(135deg, rgba(239,68,68,0.15) 0%, rgba(127,29,29,0.25) 100%);
        border: 1.5px solid #EF4444;
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 16px;
        animation: pulseAlert 1.6s ease-in-out infinite;
        box-shadow: 0 0 30px rgba(239,68,68,0.2);
        backdrop-filter: blur(10px);
    }
    .status-card-no, .status-card-clear {
        background: linear-gradient(135deg, rgba(16,185,129,0.10) 0%, rgba(6,78,59,0.20) 100%);
        border: 1.5px solid rgba(16,185,129,0.4);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 0 20px rgba(16,185,129,0.1);
        backdrop-filter: blur(10px);
    }
    .status-title {
        font-family: 'Space Mono', monospace;
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #64748B;
        margin-bottom: 8px;
    }
    .status-value-yes, .status-value-alert {
        font-family: 'Syne', sans-serif;
        font-size: 1.3rem;
        font-weight: 800;
        color: #FCA5A5;
        display: flex;
        align-items: center;
        gap: 10px;
        letter-spacing: -0.02em;
    }
    .status-value-no, .status-value-clear {
        font-family: 'Syne', sans-serif;
        font-size: 1.3rem;
        font-weight: 800;
        color: #6EE7B7;
        letter-spacing: -0.02em;
    }
    @keyframes pulseAlert {
        0%, 100% { box-shadow: 0 0 20px rgba(239,68,68,0.2); border-color: #EF4444; }
        50%       { box-shadow: 0 0 45px rgba(239,68,68,0.5); border-color: #FCA5A5; }
    }
    @keyframes pulse-border {
        0%   { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        70%  { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    /* ================================================================
       METRIC PILLS
    ================================================================ */
    .metric-pill {
        background: rgba(15,23,42,0.7);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 14px 18px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-pill:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }
    .metric-pill-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.68rem;
        color: #64748B;
        font-weight: 400;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-pill-val {
        font-family: 'Syne', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.03em;
        line-height: 1;
    }

    /* ================================================================
       MINI STAT BADGES (Sidebar)
    ================================================================ */
    .stat-mini {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 10px 12px;
        text-align: center;
        backdrop-filter: blur(5px);
    }
    .stat-mini .val {
        font-family: 'Syne', sans-serif;
        font-size: 1.15rem;
        font-weight: 800;
        color: #E2E8F0;
        line-height: 1;
    }
    .stat-mini .lbl {
        font-family: 'Space Mono', monospace;
        font-size: 0.62rem;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 3px;
    }

    /* ================================================================
       TOP-RIGHT ACCOUNT BAR
    ================================================================ */
    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0 16px 0;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 20px;
    }
    .topbar-user {
        display: flex;
        align-items: center;
        gap: 12px;
        background: rgba(15,23,42,0.8);
        border: 1px solid rgba(16,185,129,0.2);
        padding: 8px 16px;
        border-radius: 999px;
        backdrop-filter: blur(10px);
        transition: border-color 0.2s ease;
    }
    .topbar-user:hover { border-color: rgba(16,185,129,0.4); }
    .topbar-user .avatar {
        width: 32px; height: 32px; border-radius: 50%;
        background: linear-gradient(135deg, #10B981, #059669);
        display: flex; align-items: center; justify-content: center;
        font-family: 'Syne', sans-serif;
        font-weight: 800; color: white; font-size: 0.78rem;
        flex-shrink: 0;
        box-shadow: 0 0 10px rgba(16,185,129,0.4);
    }
    .topbar-user .name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.88rem;
        color: #E2E8F0;
        font-weight: 600;
        line-height: 1.2;
    }
    .topbar-user .role {
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        color: #475569;
        letter-spacing: 0.04em;
    }

    /* ================================================================
       LIVE BADGE
    ================================================================ */
    .live-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: rgba(16,185,129,0.12);
        color: #10B981;
        padding: 6px 14px;
        border-radius: 999px;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        border: 1px solid rgba(16,185,129,0.3);
        text-transform: uppercase;
    }
    .live-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        background: #10B981;
        animation: livePulse 1.2s ease-in-out infinite;
        display: inline-block;
    }
    @keyframes livePulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%       { opacity: 0.4; transform: scale(0.7); }
    }

    /* ================================================================
       SIDEBAR SECTION CAPTION
    ================================================================ */
    .sidebar-section-caption {
        font-family: 'Space Mono', monospace;
        color: #475569;
        font-size: 0.72rem;
        margin-top: -4px;
        margin-bottom: 8px;
        letter-spacing: 0.04em;
    }

    /* ================================================================
       EMPTY STATE PLACEHOLDER
    ================================================================ */
    .empty-state {
        border: 2px dashed rgba(16,185,129,0.15);
        border-radius: 18px;
        padding: 50px 30px;
        text-align: center;
        color: #475569;
        background: rgba(16,185,129,0.02);
    }
    .empty-state-icon { font-size: 3rem; margin-bottom: 12px; }
    .empty-state h4 {
        font-family: 'Syne', sans-serif;
        color: #94A3B8;
        font-weight: 700;
        margin: 0 0 8px 0;
        font-size: 1.1rem;
    }
    .empty-state p {
        font-size: 0.88rem;
        color: #475569;
        margin: 0;
        line-height: 1.5;
    }

    /* ================================================================
       GLASS CARD (generic container)
    ================================================================ */
    .glass-card {
        background: rgba(15,23,42,0.6);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        margin-bottom: 12px;
    }

    /* ================================================================
       HORIZONTAL CONTROL BAR
    ================================================================ */
    .ctrl-bar {
        background: rgba(10,18,34,0.85);
        border: 1px solid rgba(16,185,129,0.12);
        border-top: 2px solid rgba(16,185,129,0.25);
        border-radius: 14px;
        padding: 12px 20px 14px;
        margin-bottom: 18px;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04);
        position: relative;
    }
    /* Label overrides inside the control bar */
    .ctrl-bar label,
    .ctrl-bar .stSelectbox label,
    .ctrl-bar .stSlider label,
    .ctrl-bar .stCheckbox label,
    .ctrl-bar .stRadio label {
        font-family: 'Space Mono', monospace !important;
        font-size: 0.65rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.07em !important;
        color: #475569 !important;
        font-weight: 400 !important;
    }
    /* Tighten toggle row height */
    .ctrl-bar .stCheckbox, .ctrl-bar [data-testid="stToggleSwitch"] {
        margin-bottom: 2px !important;
    }
    /* Vertical divider between column groups */
    .ctrl-bar [data-testid="column"]:not(:last-child)::after {
        content: '';
        position: absolute;
        right: 0; top: 10%; height: 80%;
        width: 1px;
        background: rgba(255,255,255,0.06);
    }
    </style>
    """

