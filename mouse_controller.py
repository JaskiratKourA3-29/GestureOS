"""
╔══════════════════════════════════════════╗
║  GestureOS — Mouse Controller Module     ║
║  Smooth cursor movement via exponential  ║
║  smoothing + click with cooldown guard.  ║
╚══════════════════════════════════════════╝
"""

import pyautogui
import numpy as np
import time

pyautogui.FAILSAFE = False   # Prevent corner-exit crash during gesture control
pyautogui.PAUSE    = 0.0     # Remove built-in delay for real-time response


class MouseController:
    """
    Maps webcam hand-tip coordinates to screen coordinates with
    exponential smoothing to eliminate natural hand tremor (jitter).

    Design Decisions
    ----------------
    * A *control zone* (inner 70 % of the frame after margin stripping)
      is mapped to the full screen so the user doesn't need to reach
      frame edges to touch screen corners.
    * Exponential moving average with ALPHA = 0.18 gives ~82 % history
      weighting, producing smooth but responsive movement.
    * Click has its own internal cooldown so a slow pinch release
      doesn't fire multiple clicks.
    """

    ALPHA       = 0.18    # Smoothing factor  (lower → smoother, higher → faster)
    MARGIN      = 0.15    # Fraction to strip from each side as dead zone
    CLICK_DELAY = 0.60    # Minimum seconds between consecutive clicks

    def __init__(self) -> None:
        self.sw, self.sh  = pyautogui.size()      # Screen dimensions
        self._px = self.sw // 2                    # Previous smooth X
        self._py = self.sh // 2                    # Previous smooth Y
        self._last_click  = 0.0

    # ------------------------------------------------------------------
    def _to_screen(self, px: int, py: int, fw: int, fh: int):
        """
        Convert a webcam pixel (px, py) inside frame (fw × fh)
        to a screen pixel, stripping the dead-zone margin first.

        Returns (screen_x, screen_y)  –  both clamped to screen bounds.
        """
        # Normalise to [0, 1]
        nx = px / fw
        ny = py / fh

        # Remove margin and re-normalise so inner zone → full range
        nx = np.clip((nx - self.MARGIN) / (1.0 - 2 * self.MARGIN), 0.0, 1.0)
        ny = np.clip((ny - self.MARGIN) / (1.0 - 2 * self.MARGIN), 0.0, 1.0)

        return int(nx * self.sw), int(ny * self.sh)

    # ------------------------------------------------------------------
    def move(self, index_tip: tuple, frame_shape: tuple) -> None:
        """
        Move the system cursor, smoothed toward *index_tip*.

        Parameters
        ----------
        index_tip   : (x, y) pixel in the webcam frame.
        frame_shape : frame.shape  →  (height, width, channels).
        """
        h, w    = frame_shape[:2]
        raw_x, raw_y = self._to_screen(index_tip[0], index_tip[1], w, h)

        # Exponential moving average
        sx = int(self.ALPHA * raw_x + (1 - self.ALPHA) * self._px)
        sy = int(self.ALPHA * raw_y + (1 - self.ALPHA) * self._py)

        pyautogui.moveTo(sx, sy)
        self._px, self._py = sx, sy

    # ------------------------------------------------------------------
    def click(self) -> bool:
        """
        Fire a left mouse-click if cooldown has elapsed.

        Returns True if click was executed, False if still on cooldown.
        """
        now = time.time()
        if now - self._last_click < self.CLICK_DELAY:
            return False
        pyautogui.click()
        self._last_click = now
        return True
