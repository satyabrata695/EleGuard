"""
EleGuard Custom Design System & CSS Tokens
Includes dark glassmorphism, responsive status cards, and glowing badges.
"""


def get_custom_css() -> str:
    """Return complete CSS stylesheets for EleGuard dashboard."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Brand Header */
    .brand-container {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 16px 20px;
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .brand-logo {
        font-size: 2.8rem;
        line-height: 1;
        filter: drop-shadow(0 0 10px rgba(16, 185, 129, 0.4));
    }
    .brand-title {
        font-size: 1.9rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .brand-title span {
        color: #10B981;
    }
    .brand-subtitle {
        font-size: 0.92rem;
        color: #94A3B8;
        margin: 2px 0 0 0;
    }

    /* Status Indicator Cards */
    .status-card-yes {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.25) 100%);
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        margin-bottom: 16px;
        animation: pulse-border 2s infinite;
    }
    .status-card-no {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%);
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        margin-bottom: 16px;
    }
    .status-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #CBD5E1;
        margin-bottom: 4px;
    }
    .status-value-yes {
        font-size: 2.2rem;
        font-weight: 800;
        color: #EF4444;
        margin: 0;
    }
    .status-value-no {
        font-size: 2.2rem;
        font-weight: 800;
        color: #10B981;
        margin: 0;
    }

    /* Metrics & Pill Badges */
    .metric-pill {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-pill-label {
        font-size: 0.8rem;
        color: #94A3B8;
        font-weight: 600;
        margin-bottom: 2px;
    }
    .metric-pill-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #F8FAFC;
    }

    @keyframes pulse-border {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        70% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    </style>
    """
