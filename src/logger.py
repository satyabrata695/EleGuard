"""
Legacy Compatibility Bridge for src.logger -> backend.logger
"""
from backend.logger import Logger, get_logger

__all__ = ["Logger", "get_logger"]