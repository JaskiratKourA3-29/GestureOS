#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║   HandCanvas AI — Real-Time Air Drawing System           ║
║   Draw in the air with your finger, save your art!       ║
║                                                          ║
║  Controls                                                ║
║  ──────────────────────────────────────────────          ║
║  1 Finger  →  Draw  (move index tip to paint)            ║
║  2 Fingers →  Erase  (hover to erase)                    ║
║  3 Fingers →  Change colour  (cycles through palette)    ║
║  4 Fingers →  Change brush size                          ║
║  5 Fingers →  Save canvas as PNG                         ║
║  Pinch     →  Clear entire canvas                        ║
║  Q         →  Quit                                       ║
║                                                          ║
║  FIX: _HandTracker rewritten to use the MediaPipe        ║
║  Tasks API (compatible with MediaPipe >= 0.10.x).        ║
╚══════════════════════════════════════════════════════════╝
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import math
import time
import os
import urllib.request
from pathlib import Path
from datetime import datetime


# ─── Colour Palette (BGR) ────────────────────────────────────────────────────
COLOURS = [
    (255, 255, 255),   # White
    (0,   230, 100),   # Green
    (80,  180, 255),   # Sky Blue
    (60,   60, 255),   # Red
    (255, 180,  50),   # Orange
    (220,  80, 220),   # Purple
    (0,   220, 220),   # Cyan
    (255, 230,   0),   # Yellow
]

COLOUR_NAMES = ["White", "Green", "Blue", "Red",
                "Orange", "Purple", "Cyan", "Yellow"]

BRUSH_SIZES = [4, 8, 14, 22, 34]   # pixels


# ─── Model Auto-Download ──────────────────────────────────────────────────────
_MODEL_FILENAME = "hand_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def _ensure_model_downloaded() -> None:
    """Download the hand landmarker model if not already present (~8 MB, once)."""
    if os.path.exists(_MODEL_FILENAME):
        return

    print("\n" + "═" * 56)
    print("  Downloading MediaPipe Hand Landmark Model (~8 MB)")
    print("  This only happens once and is cached locally.")
    print("═" * 56)

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(int(downloaded * 100 / total_size), 100) if total_size > 0 else 0
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"  [{bar}] {pct}%", end="\r", flush=True)

    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_FILENAME, _progress)
        print(f"\n  ✓ Model saved → {_MODEL_FILENAME}")
        print("═" * 56 + "\n")
    except Exception as exc:
        if os.path.exists(_MODEL_FILENAME):
            os.remove(_MODEL_FILENAME)
        raise RuntimeError(
            f"\n[HandCanvas] Model download failed: {exc}\n"
            f"  Please manually download from:\n  {_MODEL_URL}\n"
            f"  and save it as:  {_MODEL_FILENAME}\n"
        )


# ─── Landmark Index Constants ─────────────────────────────────────────────────
_THUMB_TIP  =  4
_INDEX_MCP  =  5
_INDEX_PIP  =  6
_INDEX_TIP  =  8
_MIDDLE_PIP = 10
_MIDDLE_TIP = 12
_RING_PIP   = 14
_RING_TIP   = 16
_PINKY_PIP  = 18
_PINKY_TIP  = 20

_TIPS    = [_INDEX_TIP, _MIDDLE_TIP, _RING_TIP, _PINKY_TIP]
_PIPS    = [_INDEX_PIP, _MIDDLE_PIP, _RING_PIP, _PINKY_PIP]
_TIP_SET = {_THUMB_TIP, _INDEX_TIP, _MIDDLE_TIP, _RING_TIP, _PINKY_TIP}

# ─── Hand Skeleton Connections ────────────────────────────────────────────────
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


# ─── Gesture Detector (self-contained, Tasks API) ─────────────────────────────
class _HandTracker:
    """
    Self-contained hand tracker using the MediaPipe Tasks API.
    Replaces the legacy mp.solutions.hands which is unavailable
    in MediaPipe >= 0.10.x.

    Public interface is identical to the original so no other
    code in this file needs to change.
    """

    PINCH_THRESHOLD     = 0.06
    THUMB_EXTEND_THRESH = 0.06

    def __init__(self):
        _ensure_model_downloaded()

        base_opts = mp_python.BaseOptions(model_asset_path=_MODEL_FILENAME)
        lm_opts   = mp_vision.HandLandmarkerOptions(
            base_options                  = base_opts,
            running_mode                  = mp_vision.RunningMode.VIDEO,
            num_hands                     = 1,
            min_hand_detection_confidence = 0.72,
            min_hand_presence_confidence  = 0.72,
            min_tracking_confidence       = 0.60,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(lm_opts)

        # VIDEO mode requires strictly-increasing ms timestamps
        self._origin_ms = int(time.time() * 1000)
        self._last_ts   = -1

    # ── Timestamp ─────────────────────────────────────────────────────────────
    def _next_ts(self) -> int:
        now = int(time.time() * 1000) - self._origin_ms
        if now <= self._last_ts:
            now = self._last_ts + 1
        self._last_ts = now
        return now

    # ── Core detection ────────────────────────────────────────────────────────
    def process(self, bgr):
        """
        Run hand detection on a BGR OpenCV frame.
        Returns a HandLandmarkerResult with a .hand_landmarks list.
        """
        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb    = np.ascontiguousarray(rgb)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._landmarker.detect_for_video(mp_img, self._next_ts())

    # ── Gesture helpers ───────────────────────────────────────────────────────
    def count_fingers(self, lm) -> int:
        """
        lm — list of 21 NormalizedLandmark (result.hand_landmarks[0]).
        Access as lm[index], not lm.landmark[index].
        """
        count = sum(
            1 for tip, pip in zip(_TIPS, _PIPS)
            if lm[tip].y < lm[pip].y
        )
        if abs(lm[_THUMB_TIP].x - lm[_INDEX_MCP].x) > self.THUMB_EXTEND_THRESH:
            count += 1
        return count

    def pinch_dist(self, lm) -> float:
        t = lm[_THUMB_TIP]
        i = lm[_INDEX_TIP]
        return math.hypot(t.x - i.x, t.y - i.y)

    def index_pos(self, lm, h, w):
        p = lm[_INDEX_TIP]
        return int(p.x * w), int(p.y * h)

    def draw_skeleton(self, frame, results):
        """
        Draw hand skeleton directly with OpenCV.
        No mp.solutions.drawing_utils dependency.
        """
        if not results.hand_landmarks:
            return
        h, w = frame.shape[:2]
        for lms in results.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
            for a, b in _HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (60, 200, 60), 2, cv2.LINE_AA)
            for idx, pt in enumerate(pts):
                r = 6 if idx in _TIP_SET else 4
                cv2.circle(frame, pt, r, (255, 255, 255), -1)
                cv2.circle(frame, pt, r, (0,   200, 100),  1)

    def release(self):
        self._landmarker.close()


# ─── Canvas Manager ───────────────────────────────────────────────────────────
class _Canvas:
    def __init__(self, h: int, w: int):
        self._h = h
        self._w = w
        self.layer = np.zeros((h, w, 3), dtype=np.uint8)

    def draw_stroke(self, pt1, pt2, color, size):
        if pt1 and pt2:
            cv2.line(self.layer, pt1, pt2, color, size, cv2.LINE_AA)

    def erase(self, center, size):
        r = size * 3
        cv2.circle(self.layer, center, r, (0, 0, 0), -1)

    def clear(self):
        self.layer[:] = 0

    def blend_onto(self, frame: np.ndarray) -> np.ndarray:
        """Overlay canvas layer onto the webcam frame."""
        mask = cv2.cvtColor(self.layer, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        cam_part   = cv2.bitwise_and(frame,      frame,      mask=mask_inv)
        paint_part = cv2.bitwise_and(self.layer, self.layer, mask=mask)
        return cv2.add(cam_part, paint_part)

    def save(self, folder: str = "handcanvas_saves") -> str:
        Path(folder).mkdir(exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(folder, f"handcanvas_{ts}.png")
        cv2.imwrite(path, self.layer)
        return path


# ─── UI Overlay ───────────────────────────────────────────────────────────────
def _draw_ui(frame, colour_idx, brush_idx, fingers, status, is_drawing, is_erasing):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 56), (18, 18, 26), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cv2.putText(frame, "HandCanvas AI", (12, 36),
                font, 0.75, (210, 230, 255), 2, cv2.LINE_AA)

    sx = 220
    for i, col in enumerate(COLOURS):
        bx, by = sx + i * 38, 10
        cv2.rectangle(frame, (bx, by), (bx + 28, by + 28), col, -1)
        if i == colour_idx:
            cv2.rectangle(frame, (bx - 2, by - 2),
                          (bx + 30, by + 30), (255, 255, 255), 2)

    bx = sx + len(COLOURS) * 38 + 16
    cv2.putText(frame, "Brush:", (bx, 30), font, 0.40, (160, 160, 175), 1, cv2.LINE_AA)
    bx2 = bx + 55
    r = BRUSH_SIZES[brush_idx]
    cv2.circle(frame, (bx2 + r + 2, 26), r, COLOURS[colour_idx], -1)

    if is_drawing:
        badge, bc = "DRAWING", (80, 215, 90)
    elif is_erasing:
        badge, bc = "ERASING", (50, 165, 255)
    else:
        badge, bc = "HOVERING", (105, 105, 115)

    cv2.putText(frame, badge, (w - 160, 36),
                font, 0.55, bc, 2, cv2.LINE_AA)

    if status:
        ov2 = frame.copy()
        cv2.rectangle(ov2, (0, h - 40), (w, h), (18, 18, 26), -1)
        cv2.addWeighted(ov2, 0.85, frame, 0.15, 0, frame)
        cv2.putText(frame, status, (14, h - 14),
                    font, 0.50, (100, 220, 80), 1, cv2.LINE_AA)

    hints = ["1: Draw", "2: Erase", "3: Colour", "4: Brush", "5: Save",
             "Pinch: Clear", "Q: Quit"]
    px, py = w - 130, 80
    ov3 = frame.copy()
    cv2.rectangle(ov3, (px - 8, py - 18),
                  (w - 4, py + len(hints) * 20 + 4), (20, 20, 30), -1)
    cv2.addWeighted(ov3, 0.82, frame, 0.18, 0, frame)
    for i, h_txt in enumerate(hints):
        cv2.putText(frame, h_txt, (px, py + i * 20),
                    font, 0.36, (140, 140, 155), 1, cv2.LINE_AA)


# ─── Main Application ─────────────────────────────────────────────────────────
def main():
    print("\n" + "═" * 52)
    print("  HandCanvas AI — Real-Time Air Drawing System")
    print("═" * 52)
    print("  Draw in the air using your index finger!")
    print("  1 Finger  → Draw")
    print("  2 Fingers → Erase")
    print("  3 Fingers → Cycle colour")
    print("  4 Fingers → Cycle brush size")
    print("  5 Fingers → Save canvas")
    print("  Pinch     → Clear canvas")
    print("  Q         → Quit")
    print("═" * 52 + "\n")

    tracker = _HandTracker()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    ret, first = cap.read()
    if not ret:
        print("[ERROR] Cannot read from webcam.")
        return

    first   = cv2.flip(first, 1)
    fh, fw  = first.shape[:2]
    canvas  = _Canvas(fh, fw)

    colour_idx  = 0
    brush_idx   = 1
    prev_pos    = None
    status_text = "HandCanvas ready — show your hand!"
    status_at   = time.time()
    STATUS_TTL  = 2.5

    last_gesture_time = 0.0
    GESTURE_COOLDOWN  = 1.2

    is_drawing = False
    is_erasing = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        now   = time.time()

        # ── FIXED: Tasks API uses result.hand_landmarks (not multi_hand_landmarks)
        results  = tracker.process(frame)

        fingers  = -1
        cur_pos  = None
        pinching = False

        if results.hand_landmarks:                          # ← Fixed check
            lm       = results.hand_landmarks[0]           # ← Fixed access
            fingers  = tracker.count_fingers(lm)
            cur_pos  = tracker.index_pos(lm, fh, fw)
            pdist    = tracker.pinch_dist(lm)
            pinching = pdist < tracker.PINCH_THRESHOLD
            tracker.draw_skeleton(frame, results)

        # ── Drawing (1 finger) ───────────────────────────────────────────────
        is_drawing = (fingers == 1 and not pinching)
        is_erasing = (fingers == 2)

        if is_drawing and cur_pos:
            canvas.draw_stroke(prev_pos, cur_pos,
                               COLOURS[colour_idx],
                               BRUSH_SIZES[brush_idx])
        elif is_erasing and cur_pos:
            canvas.erase(cur_pos, BRUSH_SIZES[brush_idx])
            cv2.circle(frame, cur_pos,
                       BRUSH_SIZES[brush_idx] * 3,
                       (50, 50, 60), 2)

        # ── Discrete gestures (cooldown-gated) ───────────────────────────────
        elif fingers >= 3 and now - last_gesture_time > GESTURE_COOLDOWN:

            if fingers == 3:
                colour_idx = (colour_idx + 1) % len(COLOURS)
                status_text = f"Colour: {COLOUR_NAMES[colour_idx]}"
                status_at   = now
                last_gesture_time = now

            elif fingers == 4:
                brush_idx = (brush_idx + 1) % len(BRUSH_SIZES)
                status_text = f"Brush size: {BRUSH_SIZES[brush_idx]}px"
                status_at   = now
                last_gesture_time = now

            elif fingers == 5:
                path = canvas.save()
                status_text = f"Saved: {os.path.basename(path)}"
                status_at   = now
                last_gesture_time = now
                print(f"[HandCanvas] Canvas saved → {path}")

        # ── Pinch = Clear canvas ──────────────────────────────────────────────
        if pinching and now - last_gesture_time > GESTURE_COOLDOWN:
            canvas.clear()
            status_text = "Canvas cleared!"
            status_at   = now
            last_gesture_time = now
            prev_pos = None

        prev_pos = cur_pos if (is_drawing and cur_pos) else None

        if now - status_at > STATUS_TTL:
            status_text = ""

        combined = canvas.blend_onto(frame)

        if cur_pos and fingers == 1:
            col = COLOURS[colour_idx]
            cv2.circle(combined, cur_pos, BRUSH_SIZES[brush_idx], col, -1)
            cv2.circle(combined, cur_pos, BRUSH_SIZES[brush_idx] + 2,
                       (255, 255, 255), 1)

        _draw_ui(combined, colour_idx, brush_idx,
                 fingers, status_text, is_drawing, is_erasing)

        cv2.imshow("HandCanvas AI — Air Drawing System", combined)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if canvas.layer.any():
        path = canvas.save()
        print(f"[HandCanvas] Auto-saved on exit → {path}")

    cap.release()
    cv2.destroyAllWindows()
    tracker.release()
    print("[HandCanvas] Session ended. Your art is saved!")


if __name__ == "__main__":
    main()
