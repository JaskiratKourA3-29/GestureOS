"""
╔══════════════════════════════════════════╗
║  GestureOS — UI Renderer & Utilities    ║
║  Professional dark overlay dashboard    ║
║  rendered directly onto the OpenCV      ║
║  webcam frame.                          ║
╚══════════════════════════════════════════╝

Panel Layout (right side):
  ┌───────────────────┐
  │  GESTURE  OS      │ ← header
  │ ─────────────     │
  │ ● LIVE   30 FPS  │ ← status dot + fps
  │ [████████░░]      │ ← fps bar
  │ ─────────────     │
  │ FINGERS           │
  │ 2  ○●●○○          │ ← count + pip indicators
  │ ─────────────     │
  │ GESTURE           │
  │ Take Screenshot   │
  │ ─────────────     │
  │ MODE              │
  │ Screenshot        │
  │ ─────────────     │
  │ STATUS            │
  │ Saved: abc.png    │
  └───────────────────┘

Bottom bar (left of panel):
  GESTURE MAP: 0:Lock  1:Mouse  2:Shot  3:VSCode  4:Chrome  5:Media
"""

import cv2
import numpy as np
import time
from typing import Optional


# ─── FPS Counter ──────────────────────────────────────────────────────────────
class FPSCounter:
    """
    Rolling-window FPS calculator.
    Stores the last WINDOW frame timestamps and computes average FPS
    to avoid single-frame jitter.
    """
    WINDOW = 30

    def __init__(self) -> None:
        self._ts : list[float] = []

    def tick(self) -> int:
        """Call once per frame. Returns integer FPS."""
        now = time.time()
        self._ts.append(now)
        if len(self._ts) > self.WINDOW:
            self._ts.pop(0)
        if len(self._ts) < 2:
            return 0
        elapsed = self._ts[-1] - self._ts[0]
        return int((len(self._ts) - 1) / elapsed) if elapsed > 0 else 0


# ─── Color Palette (BGR) ──────────────────────────────────────────────────────
class _C:
    BG          = (12,  12,  18)
    PANEL       = (22,  22,  32)
    TEXT        = (215, 215, 220)
    DIM         = (105, 105, 115)
    ACCENT      = (100, 220,  80)    # green
    WARN        = ( 50, 165, 255)    # orange
    DANGER      = ( 55,  55, 210)    # red
    HEADER      = (200, 230, 255)    # light blue-white
    FIN_ON      = ( 80, 215,  90)    # active finger pip
    FIN_OFF     = ( 45,  45,  58)    # inactive finger pip

    # Per-gesture accent colours (BGR)
    GESTURE_COLORS: dict[int, tuple] = {
        -1: (105, 105, 115),   # dim – no hand
         0: ( 55,  55, 215),   # red  – lock
         1: ( 80, 215,  90),   # green – mouse
         2: (  0, 185, 215),   # yellow/gold – screenshot
         3: (230, 160,  80),   # blue – vscode
         4: ( 15, 185, 255),   # orange – chrome
         5: (195,  85, 205),   # purple – media
    }


# ─── Renderer ─────────────────────────────────────────────────────────────────
class UIRenderer:
    """
    Draws the GestureOS overlay directly onto the webcam frame (in-place).
    Call ``render()`` every frame.
    """

    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    PANEL_W    = 262
    PANEL_PAD  = 14

    GESTURE_MAP = [
        ("0", "Lock"),
        ("1", "Mouse"),
        ("2", "Screenshot"),
        ("3", "VS Code"),
        ("4", "Chrome"),
        ("5", "Media"),
    ]

    # ------------------------------------------------------------------
    def _alpha_rect(self, img: np.ndarray,
                    x1: int, y1: int, x2: int, y2: int,
                    color: tuple, alpha: float = 0.88,
                    radius: int = 10) -> None:
        """Draw a semi-transparent filled rounded rectangle."""
        overlay = img.copy()
        r = radius
        # Fill body
        cv2.rectangle(overlay, (x1 + r, y1),     (x2 - r, y2),     color, -1)
        cv2.rectangle(overlay, (x1,     y1 + r), (x2,     y2 - r), color, -1)
        # Corner circles
        for cx, cy in [(x1+r, y1+r), (x2-r, y1+r),
                       (x1+r, y2-r), (x2-r, y2-r)]:
            cv2.circle(overlay, (cx, cy), r, color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    def _text(self, img: np.ndarray, txt: str,
              x: int, y: int,
              color: tuple  = None,
              scale: float  = 0.46,
              thick: int    = 1) -> None:
        cv2.putText(img, txt, (x, y), self.FONT,
                    scale, color or _C.TEXT, thick, cv2.LINE_AA)

    def _divider(self, img: np.ndarray, x1: int, x2: int, y: int) -> None:
        cv2.line(img, (x1, y), (x2, y), (55, 55, 68), 1)

    def _finger_pips(self, img: np.ndarray,
                     x: int, y: int, count: int) -> None:
        """Draw 5 small circles (filled = extended finger)."""
        r, gap = 8, 20
        for i in range(5):
            c = _C.FIN_ON if i < count else _C.FIN_OFF
            cx = x + i * gap
            cv2.circle(img, (cx, y), r, c, -1)
            cv2.circle(img, (cx, y), r, (62, 62, 75), 1)

    def _fps_bar(self, img: np.ndarray,
                 x: int, y: int, w: int, fps: int) -> None:
        fill = int(min(fps, 30) / 30 * w)
        color = _C.ACCENT if fps >= 25 else _C.WARN if fps >= 14 else _C.DANGER
        cv2.rectangle(img, (x, y-7), (x + w, y), (38, 38, 48), -1)
        if fill > 0:
            cv2.rectangle(img, (x, y-7), (x + fill, y), color, -1)

    # ------------------------------------------------------------------
    def render(self,
               frame     : np.ndarray,
               gesture,                  # GestureResult
               mode      : str,
               status    : str,
               fps       : int) -> None:
        """
        Draw the complete GestureOS overlay onto *frame* in-place.

        Parameters
        ----------
        frame   : BGR OpenCV frame (will be modified).
        gesture : GestureResult from gesture_detector.
        mode    : Current mode string  (e.g. "Mouse Mode").
        status  : Latest action status (e.g. "Saved: gesture_....png").
        fps     : Current FPS integer.
        """
        fh, fw = frame.shape[:2]

        px1 = fw - self.PANEL_W - 10
        py1 = 10
        px2 = fw - 10
        py2 = fh - 10

        # ── Right-side panel ────────────────────────────────────────────────
        self._alpha_rect(frame, px1, py1, px2, py2, _C.PANEL, alpha=0.90)

        lx  = px1 + self.PANEL_PAD       # left content X
        rx  = px2 - self.PANEL_PAD       # right edge X (for right-aligned text)
        row = py1 + 30                   # current Y cursor
        lh  = 27                         # base line-height

        # Header
        self._text(frame, "GESTURE  OS", lx, row, _C.HEADER, 0.60, 2)
        row += 10
        self._divider(frame, lx, rx, row);  row += lh - 4

        # Live dot + FPS
        dot_c = _C.ACCENT if fps >= 25 else _C.WARN
        cv2.circle(frame, (lx + 6, row - 6), 5, dot_c, -1)
        self._text(frame, "LIVE",           lx + 16, row, dot_c, 0.40)
        self._text(frame, f"{fps} FPS",     rx - 48,  row, dot_c, 0.40)
        row += 6
        self._fps_bar(frame, lx, row + lh - 12, rx - lx, fps)
        row += lh + 4
        self._divider(frame, lx, rx, row);  row += lh - 4

        # Fingers
        self._text(frame, "FINGERS", lx, row, _C.DIM, 0.36)
        row += 20
        fc   = max(gesture.fingers, 0) if gesture.hand_detected else 0
        self._text(frame, str(fc), lx, row, _C.TEXT, 1.0, 2)
        self._finger_pips(frame, lx + 32, row - 8, fc)
        row += lh + 4
        self._divider(frame, lx, rx, row);  row += lh - 4

        # Gesture label
        self._text(frame, "GESTURE", lx, row, _C.DIM, 0.36)
        row += 20
        key  = gesture.fingers if gesture.hand_detected else -1
        gc   = _C.GESTURE_COLORS.get(key, _C.DIM)
        lbl  = (gesture.label if gesture.hand_detected else "No Hand")[:22]
        self._text(frame, lbl, lx, row, gc, 0.44)
        row += lh + 4
        self._divider(frame, lx, rx, row);  row += lh - 4

        # Mode
        self._text(frame, "MODE", lx, row, _C.DIM, 0.36)
        row += 20
        self._text(frame, mode[:22], lx, row, _C.TEXT, 0.43)
        row += lh + 4
        self._divider(frame, lx, rx, row);  row += lh - 4

        # Status
        self._text(frame, "STATUS", lx, row, _C.DIM, 0.36)
        row += 20
        if status:
            # May be long filename — wrap at 22 chars
            s1 = status[:22]
            s2 = status[22:44]
            self._text(frame, s1, lx, row, _C.ACCENT, 0.40)
            if s2:
                self._text(frame, s2, lx, row + 16, _C.ACCENT, 0.40)
        else:
            self._text(frame, "Waiting...", lx, row, _C.DIM, 0.40)
        row += lh + (16 if status and len(status) > 22 else 0)

        # Pinch badge
        if gesture.pinching:
            row += 6
            self._divider(frame, lx, rx, row);  row += lh - 8
            cv2.circle(frame, (lx + 7, row - 5), 6, _C.ACCENT, -1)
            self._text(frame, "PINCH ACTIVE", lx + 18, row, _C.ACCENT, 0.36)

        # ── Index-tip cursor ring (Mouse Mode only) ──────────────────────────
        if gesture.hand_detected and gesture.fingers == 1 and gesture.index_tip:
            ix, iy = gesture.index_tip
            cv2.circle(frame, (ix, iy), 16, _C.ACCENT,  2)
            cv2.circle(frame, (ix, iy),  4, _C.ACCENT, -1)

        # ── Bottom gesture-map bar ───────────────────────────────────────────
        bar_h  = 48
        bar_x1 = 10
        bar_y1 = fh - bar_h - 8
        bar_x2 = px1 - 10
        bar_y2 = fh - 8

        self._alpha_rect(frame, bar_x1, bar_y1, bar_x2, bar_y2,
                         _C.PANEL, alpha=0.87, radius=8)

        bx = bar_x1 + 14
        by = bar_y1 + 17
        self._text(frame, "GESTURE MAP:", bx, by, _C.DIM, 0.37)

        slot_x = bx + 108
        for num, lbl in self.GESTURE_MAP:
            is_active = (gesture.hand_detected and
                         str(gesture.fingers) == num)
            color = _C.GESTURE_COLORS.get(int(num), _C.DIM)
            self._text(frame,
                       f"{num}:{lbl}",
                       slot_x, by,
                       color if is_active else _C.DIM,
                       0.37 if not is_active else 0.40,
                       1 if not is_active else 2)
            slot_x += 88

        self._text(frame,
                   "Pinch = Click  |  Q = Quit",
                   bx + 108, by + 20, _C.DIM, 0.34)

        # ── Watermark ────────────────────────────────────────────────────────
        self._text(frame, "GestureOS v1.0", 14, 28, _C.DIM, 0.42)
