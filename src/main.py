"""
Legacy Compatibility Bridge for src.main -> backend.main
"""
from backend.main import ElephantDetectionApp, create_parser, main

if __name__ == "__main__":
    raise SystemExit(main())