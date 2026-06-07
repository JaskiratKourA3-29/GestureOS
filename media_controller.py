"""
╔══════════════════════════════════════════╗
║  GestureOS — Media Controller Module    ║
║  System media key simulation via        ║
║  PyAutoGUI keyboard shortcuts.          ║
╚══════════════════════════════════════════╝
"""

import pyautogui
import time


class MediaController:
    """
    Sends OS-level media key presses (Play/Pause, Volume, Track).

    These virtual keys are handled by the OS media layer and work
    with Spotify, YouTube (browser), Windows Media Player, VLC, etc.

    All methods are guarded by ACTION_COOLDOWN to prevent rapid-fire
    accidental triggers during a held gesture.
    """

    ACTION_COOLDOWN = 0.8   # seconds

    def __init__(self) -> None:
        self._last_action = 0.0

    def _guard(self) -> bool:
        """Return True (and update timestamp) if cooldown has elapsed."""
        now = time.time()
        if now - self._last_action < self.ACTION_COOLDOWN:
            return False
        self._last_action = now
        return True

    # ── Playback ──────────────────────────────────────────────────────────────
    def play_pause(self) -> bool:
        if not self._guard(): return False
        pyautogui.press("playpause")
        print("[Media] Play / Pause")
        return True

    # ── Volume ────────────────────────────────────────────────────────────────
    def volume_up(self) -> bool:
        if not self._guard(): return False
        pyautogui.press("volumeup")
        print("[Media] Volume Up")
        return True

    def volume_down(self) -> bool:
        if not self._guard(): return False
        pyautogui.press("volumedown")
        print("[Media] Volume Down")
        return True

    def mute(self) -> bool:
        if not self._guard(): return False
        pyautogui.press("volumemute")
        print("[Media] Mute toggled")
        return True

    # ── Track Navigation ─────────────────────────────────────────────────────
    def next_track(self) -> bool:
        if not self._guard(): return False
        pyautogui.press("nexttrack")
        print("[Media] Next Track")
        return True

    def prev_track(self) -> bool:
        if not self._guard(): return False
        pyautogui.press("prevtrack")
        print("[Media] Previous Track")
        return True
