#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║        GestureOS — AI Powered Touchless Desktop          ║
║        Control System   v1.0                             ║
║                                                          ║
║  Gesture Map                                             ║
║  ──────────────────────────────────────────────────      ║
║  0 Fingers  →  Lock Screen                               ║
║  1 Finger   →  Mouse Mode  (pinch to click)              ║
║  2 Fingers  →  Take Screenshot                           ║
║  3 Fingers  →  Open VS Code                              ║
║  4 Fingers  →  Open Chrome                               ║
║  5 Fingers  →  Play / Pause Media                        ║
║  Pinch      →  Left Mouse Click  (only in Mouse Mode)    ║
║                                                          ║
║  Press  Q  to quit at any time.                          ║
╚══════════════════════════════════════════════════════════╝
"""

import cv2
import sys
import time

from gesture_detector  import GestureDetector
from mouse_controller  import MouseController
from screenshot_manager import ScreenshotManager
from app_launcher      import AppLauncher
from media_controller  import MediaController
from utils             import UIRenderer, FPSCounter


# ─── Runtime Configuration ────────────────────────────────────────────────────
WEBCAM_INDEX     = 0        # Change to 1, 2 … for external webcams
FRAME_WIDTH      = 1280
FRAME_HEIGHT     = 720
TARGET_FPS       = 30

# Cooldown: minimum seconds between discrete gesture triggers
GESTURE_COOLDOWN = 1.5

# Mouse update rate: seconds between cursor position updates (~50 Hz)
MOUSE_RATE       = 0.02


# ─── Gesture → Action Map ────────────────────────────────────────────────────
ACTIONS: dict[int, str] = {
    0: "lock_screen",
    1: "mouse_mode",        # handled separately (continuous)
    2: "screenshot",
    3: "open_vscode",
    4: "open_chrome",
    5: "media_play_pause",
}


# ─── Entry Point ─────────────────────────────────────────────────────────────
def main() -> None:

    # ── Welcome banner ────────────────────────────────────────────────────────
    print("\n" + "═" * 58)
    print("  GestureOS — AI Powered Touchless Desktop Control")
    print("═" * 58)
    print("  0 fingers  →  Lock Screen")
    print("  1 finger   →  Mouse Mode  (pinch = click)")
    print("  2 fingers  →  Take Screenshot")
    print("  3 fingers  →  Open VS Code")
    print("  4 fingers  →  Open Chrome")
    print("  5 fingers  →  Play / Pause Media")
    print("  Press Q to quit.")
    print("═" * 58 + "\n")

    # ── Init modules ──────────────────────────────────────────────────────────
    detector    = GestureDetector()
    mouse       = MouseController()
    screenshotter = ScreenshotManager()
    launcher    = AppLauncher()
    media       = MediaController()
    ui          = UIRenderer()
    fps_counter = FPSCounter()

    # ── Open webcam ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam at index {WEBCAM_INDEX}.")
        print("        Try changing WEBCAM_INDEX to 1 or 2.")
        sys.exit(1)

    print("[GestureOS] Camera opened. Show your hand to begin.\n")

    # ── State variables ───────────────────────────────────────────────────────
    last_gesture_at : float = 0.0   # timestamp of last discrete trigger
    last_mouse_at   : float = 0.0   # timestamp of last mouse position update
    current_mode    : str   = "Idle"
    status_text     : str   = "System Ready"
    status_at       : float = time.time()
    STATUS_TTL      : float = 2.5   # seconds the status text stays visible

    # ── Main loop ────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame. Exiting.")
            break

        frame = cv2.flip(frame, 1)  # Mirror so hand moves feel natural
        now   = time.time()
        fps   = fps_counter.tick()

        # ── Detection ─────────────────────────────────────────────────────────
        gesture, mp_results = detector.process(frame)
        detector.draw_landmarks(frame, mp_results)

        # ── Mouse Mode  (continuous, no discrete cooldown) ────────────────────
        if gesture.hand_detected and gesture.fingers == 1:
            current_mode = "Mouse Mode"

            # Update cursor at MOUSE_RATE frequency
            if now - last_mouse_at >= MOUSE_RATE and gesture.index_tip:
                mouse.move(gesture.index_tip, frame.shape)
                last_mouse_at = now

            # Pinch = click (only in Mouse Mode to avoid false fist-triggers)
            if gesture.pinching and mouse.click():
                status_text = "Mouse Click!"
                status_at   = now

        # ── Discrete Gesture Actions (cooldown-gated) ─────────────────────────
        elif gesture.hand_detected and now - last_gesture_at >= GESTURE_COOLDOWN:

            action = ACTIONS.get(gesture.fingers)

            # 0 fingers — Lock Screen
            if action == "lock_screen":
                current_mode         = "Lock Screen"
                status_text          = launcher.lock_screen()
                status_at            = now
                last_gesture_at      = now

            # 2 fingers — Screenshot
            elif action == "screenshot":
                current_mode         = "Screenshot"
                fname                = screenshotter.capture()
                status_text          = f"Saved: {fname}"
                status_at            = now
                last_gesture_at      = now

            # 3 fingers — VS Code
            elif action == "open_vscode":
                current_mode         = "VS Code"
                status_text          = launcher.launch_vscode()
                status_at            = now
                last_gesture_at      = now

            # 4 fingers — Chrome
            elif action == "open_chrome":
                current_mode         = "Chrome"
                status_text          = launcher.launch_chrome()
                status_at            = now
                last_gesture_at      = now

            # 5 fingers — Media Play/Pause
            elif action == "media_play_pause":
                current_mode         = "Media Control"
                media.play_pause()
                status_text          = "Play / Pause"
                status_at            = now
                last_gesture_at      = now

        elif not gesture.hand_detected:
            current_mode = "Idle"

        # ── Expire status text ────────────────────────────────────────────────
        if now - status_at > STATUS_TTL:
            status_text = ""

        # ── Render overlay ────────────────────────────────────────────────────
        ui.render(frame, gesture, current_mode, status_text, fps)

        # ── Show window ───────────────────────────────────────────────────────
        cv2.imshow("GestureOS — Touchless Desktop Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[GestureOS] Q pressed — shutting down.")
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    detector.release()
    print("[GestureOS] Session ended. Goodbye!")


if __name__ == "__main__":
    main()
