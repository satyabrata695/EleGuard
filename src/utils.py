"""
Legacy Compatibility Bridge for src.utils -> backend.utils
"""
from backend.utils import FPSCounter, create_project_directories, generate_filename, get_system_info

__all__ = ["FPSCounter", "create_project_directories", "generate_filename", "get_system_info"]