"""
╔══════════════════════════════════════════╗
║  GestureOS — App Launcher Module        ║
║  Cross-platform launch: VS Code,        ║
║  Chrome, and screen lock.               ║
╚══════════════════════════════════════════╝
"""

import os
import platform
import subprocess
import time


class AppLauncher:
    """
    Launches desktop applications via gesture commands.

    Platform Support
    ----------------
    Windows  : ``subprocess.Popen`` / ``os.system``
    macOS    : ``open -a <App Name>``
    Linux    : Standard binary names on PATH

    Duplicate-Launch Guard
    ----------------------
    Each app tracks its last-launch timestamp.  Re-launching within
    LAUNCH_COOLDOWN seconds is silently ignored and a friendly message
    is returned so the UI can display it.
    """

    LAUNCH_COOLDOWN = 4.0   # seconds between consecutive app opens

    def __init__(self) -> None:
        self._platform    = platform.system()   # 'Windows' | 'Darwin' | 'Linux'
        self._last_launch : dict[str, float] = {}

    # ── Internal helpers ─────────────────────────────────────────────────────
    def _ready(self, key: str) -> bool:
        return time.time() - self._last_launch.get(key, 0.0) > self.LAUNCH_COOLDOWN

    def _stamp(self, key: str) -> None:
        self._last_launch[key] = time.time()

    def _run(self, *args, shell: bool = False) -> None:
        """Launch a process without blocking the gesture loop."""
        subprocess.Popen(list(args), shell=shell,
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)

    # ── VS Code ───────────────────────────────────────────────────────────────
    def launch_vscode(self) -> str:
        """Open Visual Studio Code. Returns a status string for the UI."""
        if not self._ready("vscode"):
            return "VS Code: already opening..."

        try:
            if   self._platform == "Windows": self._run("code", shell=True)
            elif self._platform == "Darwin":  self._run("open", "-a", "Visual Studio Code")
            else:                             self._run("code")

            self._stamp("vscode")
            print("[Launcher] VS Code launched")
            return "VS Code Opened!"

        except FileNotFoundError:
            print("[Launcher] VS Code not found on PATH")
            return "VS Code not installed"
        except Exception as exc:
            print(f"[Launcher] VS Code error: {exc}")
            return "VS Code failed"

    # ── Google Chrome ─────────────────────────────────────────────────────────
    def launch_chrome(self) -> str:
        """Open Google Chrome. Returns a status string for the UI."""
        if not self._ready("chrome"):
            return "Chrome: already opening..."

        try:
            if   self._platform == "Windows": self._run("start", "chrome", shell=True)
            elif self._platform == "Darwin":  self._run("open", "-a", "Google Chrome")
            else:                             self._run("google-chrome")

            self._stamp("chrome")
            print("[Launcher] Chrome launched")
            return "Chrome Opened!"

        except FileNotFoundError:
            print("[Launcher] Chrome not found on PATH")
            return "Chrome not installed"
        except Exception as exc:
            print(f"[Launcher] Chrome error: {exc}")
            return "Chrome failed"

    # ── Screen Lock ───────────────────────────────────────────────────────────
    def lock_screen(self) -> str:
        """Lock the workstation. Returns a status string for the UI."""
        try:
            if   self._platform == "Windows":
                os.system("rundll32.exe user32.dll,LockWorkStation")
            elif self._platform == "Darwin":
                os.system("pmset displaysleepnow")
            else:
                # Try common Linux lockers in order
                os.system(
                    "gnome-screensaver-command -l "
                    "|| loginctl lock-session "
                    "|| xdg-screensaver lock"
                )
            print("[Launcher] Screen locked")
            return "Screen Locked"

        except Exception as exc:
            print(f"[Launcher] Lock screen error: {exc}")
            return "Lock failed"
