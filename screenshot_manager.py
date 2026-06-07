"""
╔══════════════════════════════════════════╗
║  GestureOS — Screenshot Manager Module  ║
║  Timestamped screenshot capture with    ║
║  duplicate-prevention cooldown.         ║
╚══════════════════════════════════════════╝
"""

import pyautogui
import time
from pathlib import Path


class ScreenshotManager:
    """
    Captures full-screen screenshots and saves them to ./screenshots/
    with ISO-8601-style timestamps so files sort chronologically.

    Usage
    -----
    >>> mgr = ScreenshotManager()
    >>> filename = mgr.capture()          # Returns 'gesture_20250610_143022.png'
    """

    OUTPUT_DIR      = Path("screenshots")
    PREFIX          = "gesture_"
    CAPTURE_COOLDOWN = 2.0    # seconds — prevents duplicate saves per gesture hold

    def __init__(self) -> None:
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self._last_capture = 0.0

    # ------------------------------------------------------------------
    def capture(self) -> str:
        """
        Take a screenshot and save it.

        Returns
        -------
        str
            Filename only (e.g. ``gesture_20250610_143022.png``) on success,
            or ``'Too soon!'`` if called within the cooldown window.
        """
        now = time.time()
        if now - self._last_capture < self.CAPTURE_COOLDOWN:
            return "Too soon!"

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename  = f"{self.PREFIX}{timestamp}.png"
        filepath  = self.OUTPUT_DIR / filename

        image = pyautogui.screenshot()
        image.save(str(filepath))

        self._last_capture = now
        print(f"[Screenshot] Saved → {filepath}")
        return filename
