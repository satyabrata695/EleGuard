"""
Legacy Compatibility Bridge for src.dashboard -> frontend.app
"""
from frontend.app import EleGuardApp

if __name__ == "__main__":
    app = EleGuardApp()
    app.run()