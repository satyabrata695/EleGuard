"""
Legacy Compatibility Bridge for src.alerts -> backend.alerts
"""
from backend.alerts import Alert, AlertManager

__all__ = ["Alert", "AlertManager"]