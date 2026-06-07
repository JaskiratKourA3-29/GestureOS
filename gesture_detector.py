"""
╔══════════════════════════════════════════╗
║  GestureOS — Gesture Detector Module     ║
║  MediaPipe Tasks API (0.10.x compatible) ║
║  Finger counting & pinch detection.      ║
╚══════════════════════════════════════════╝

FIX: Rewritten from mp.solutions.hands (legacy) to the
     MediaPipe Tasks API, which is the only API available
     in MediaPipe >= 0.10.x installations.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import math
import numpy as np
import time
import os
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple


# ─── Model Auto-Download ──────────────────────────────────────────────────────
_MODEL_FILENAME = "hand_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def _ensure_model_downloaded() -> None:
    """
    Download the hand landmarker model file if it is not already present.
    The file is ~8 MB and is saved once to the project directory.
    """
    if os.path.exists(_MODEL_FILENAME):
        return

    print("\n" + "═" * 56)
    print("  Downloading MediaPipe Hand Landmark Model (~8 MB)")
    print("  This only happens once and is cached locally.")
    print("═" * 56)

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
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
            os.remove(_MODEL_FILENAME)          # Remove partial download
        raise RuntimeError(
            f"\n[GestureOS] Model download failed: {exc}\n"
            f"  Please manually download from:\n"
            f"  {_MODEL_URL}\n"
            f"  and save it as:  {_MODEL_FILENAME}\n"
        )


# ─── Landmark Index Constants ─────────────────────────────────────────────────
#   MediaPipe hand topology — 21 landmarks per hand:
#   Wrist=0, Thumb TIP=4, Index(MCP=5, PIP=6, TIP=8),
#   Middle(PIP=10, TIP=12), Ring(PIP=14, TIP=16), Pinky(PIP=18, TIP=20)

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

# ─── Hand Skeleton Connections (for manual OpenCV drawing) ────────────────────
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),      # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),    # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),    # Pinky
    (5, 9), (9, 13), (13, 17),                # Palm knuckle line
]

# ─── Gesture Labels ──────────────────────────────────────────────────────────
GESTURE_NAMES: dict[int, str] = {
    -1: "No Hand Detected",
     0: "Lock Screen",
     1: "Mouse Mode",
     2: "Take Screenshot",
     3: "Open VS Code",
     4: "Open Chrome",
     5: "Media Control",
}


# ─── Data Class ──────────────────────────────────────────────────────────────
@dataclass
class GestureResult:
    """
    Structured result returned from a single frame of gesture detection.

    Attributes
    ----------
    fingers       : Number of extended fingers (0-5), -1 if no hand found.
    label         : Human-readable gesture name.
    pinching      : True when thumb-tip to index-tip distance < threshold.
    pinch_dist    : Normalised Euclidean distance between the two tips.
    index_tip     : Pixel (x, y) of the index-finger tip in the frame.
    hand_detected : Whether a hand was found in this frame.
    """
    fingers      : int                        = -1
    label        : str                        = "No Hand Detected"
    pinching     : bool                       = False
    pinch_dist   : float                      = 1.0
    index_tip    : Optional[Tuple[int, int]]  = None
    hand_detected: bool                       = False


# ─── Detector Class ───────────────────────────────────────────────────────────
class GestureDetector:
    """
    Wraps MediaPipe Tasks HandLandmarker to provide per-frame gesture data.
    Compatible with MediaPipe 0.10.x (Tasks API only — no mp.solutions needed).

    Usage
    -----
    >>> detector = GestureDetector()        # auto-downloads model on first run
    >>> gesture, raw = detector.process(bgr_frame)
    >>> detector.draw_landmarks(frame, raw)
    >>> detector.release()
    """

    # Tuneable thresholds
    PINCH_THRESHOLD     = 0.055   # Normalised distance (0–1)
    THUMB_EXTEND_THRESH = 0.06    # Normalised x-distance for thumb extension
    DETECT_CONFIDENCE   = 0.75
    TRACKING_CONFIDENCE = 0.60

    def __init__(self) -> None:
        _ensure_model_downloaded()

        base_opts = mp_python.BaseOptions(model_asset_path=_MODEL_FILENAME)
        lm_opts   = mp_vision.HandLandmarkerOptions(
            base_options                  = base_opts,
            running_mode                  = mp_vision.RunningMode.VIDEO,
            num_hands                     = 1,
            min_hand_detection_confidence = self.DETECT_CONFIDENCE,
            min_hand_presence_confidence  = self.DETECT_CONFIDENCE,
            min_tracking_confidence       = self.TRACKING_CONFIDENCE,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(lm_opts)

        # VIDEO mode requires strictly-increasing millisecond timestamps
        self._origin_ms = int(time.time() * 1000)
        self._last_ts   = -1

    # ── Timestamp helpers ─────────────────────────────────────────────────────
    def _next_ts(self) -> int:
        """Return a strictly-increasing ms timestamp (required by VIDEO mode)."""
        now = int(time.time() * 1000) - self._origin_ms
        if now <= self._last_ts:
            now = self._last_ts + 1
        self._last_ts = now
        return now

    # ── Gesture maths ─────────────────────────────────────────────────────────
    def _count_fingers(self, lms) -> int:
        """
        Count extended fingers.
        lms — list of 21 NormalizedLandmark objects from the Tasks API.
        Index to Pinky: tip.y < pip.y  → extended (y grows downward).
        Thumb: |tip.x - index_mcp.x| > threshold → extended.
        """
        count = sum(
            1 for tip_i, pip_i in zip(_TIPS, _PIPS)
            if lms[tip_i].y < lms[pip_i].y
        )
        if abs(lms[_THUMB_TIP].x - lms[_INDEX_MCP].x) > self.THUMB_EXTEND_THRESH:
            count += 1
        return count

    def _pinch_dist(self, lms) -> float:
        """Normalised Euclidean distance between thumb tip and index tip."""
        t = lms[_THUMB_TIP]
        i = lms[_INDEX_TIP]
        return math.hypot(t.x - i.x, t.y - i.y)

    # ── Public API ────────────────────────────────────────────────────────────
    def process(self, bgr_frame: np.ndarray) -> Tuple[GestureResult, object]:
        """
        Detect hand in *bgr_frame* and return (GestureResult, raw_results).

        Parameters
        ----------
        bgr_frame : np.ndarray
            BGR frame from OpenCV (will not be modified).

        Returns
        -------
        (GestureResult, mediapipe HandLandmarkerResult)
        """
        # Tasks API expects an RGB image as mediapipe.Image
        rgb    = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb    = np.ascontiguousarray(rgb)     # Ensure C-contiguous memory
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect_for_video(mp_img, self._next_ts())

        if not result.hand_landmarks:
            return GestureResult(), result

        lms     = result.hand_landmarks[0]     # List of 21 NormalizedLandmark
        h, w    = bgr_frame.shape[:2]
        fingers = self._count_fingers(lms)
        dist    = self._pinch_dist(lms)
        tip     = lms[_INDEX_TIP]

        return GestureResult(
            fingers       = fingers,
            label         = GESTURE_NAMES.get(fingers, "Unknown"),
            pinching      = dist < self.PINCH_THRESHOLD,
            pinch_dist    = dist,
            index_tip     = (int(tip.x * w), int(tip.y * h)),
            hand_detected = True,
        ), result

    def draw_landmarks(self, frame: np.ndarray, results) -> None:
        """
        Draw hand skeleton and landmark dots on *frame* in-place.
        Pure OpenCV implementation — no mp.solutions.drawing_utils needed.
        """
        if not results.hand_landmarks:
            return

        h, w = frame.shape[:2]
        for lms in results.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

            # Draw bone connections
            for a, b in _HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (60, 210, 60), 2, cv2.LINE_AA)

            # Draw landmark dots (tips are larger)
            for idx, pt in enumerate(pts):
                r = 6 if idx in _TIP_SET else 4
                cv2.circle(frame, pt, r, (255, 255, 255), -1)
                cv2.circle(frame, pt, r, (0,   210, 100),  1)

    def release(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()
