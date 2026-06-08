<div align="center">

# 🖐️ GestureOS

### AI-Powered Touchless Desktop Control System

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-FF6F00?style=flat-square&logo=google&logoColor=white)](https://mediapipe.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)]()

_Control your entire desktop — mouse, apps, screenshots, media — using only hand gestures and a standard webcam. No touch required._

[Demo](#-demo) · [Features](#-features) · [Installation](#-installation) · [Gesture Map](#-gesture-map) · [Architecture](#-architecture)

---

![GestureOS Demo Placeholder](assets/demo_placeholder.png)

<img width="1918" height="1025" alt="image" src="https://github.com/user-attachments/assets/a9fba7d7-ecbd-492b-a9fa-7d8f00b56e9b" />


</div>

---

## 🎯 What Is GestureOS?

GestureOS is a **real-time AI desktop controller** that lets you operate your computer touchlessly through hand gestures. It uses computer vision and machine learning (via MediaPipe) to detect hand landmarks at 25–30 FPS and map specific finger configurations to OS-level actions.

Built as a professional portfolio project demonstrating:

- **Computer Vision** — Real-time landmark detection with MediaPipe
- **Human Computer Interaction (HCI)** — Gesture-to-action mapping with natural UX
- **Desktop Automation** — OS-level control via PyAutoGUI
- **Software Engineering** — Clean modular architecture, ready for extension

---

## ✨ Features

| Category          | Feature                                                  |
| ----------------- | -------------------------------------------------------- |
| 🖱️ Mouse Control  | Move cursor with your index finger — smooth, jitter-free |
| 👆 Click          | Pinch thumb + index finger to left-click                 |
| 📸 Screenshot     | Capture & auto-save with timestamp filename              |
| 🔒 Lock Screen    | Cross-platform screen lock (Windows / macOS / Linux)     |
| 💻 Launch Apps    | Open VS Code and Chrome hands-free                       |
| 🎵 Media Control  | Play/Pause any system media                              |
| 📊 Live HUD       | Professional dark overlay with FPS, gesture name, mode   |
| ⚡ Real-time      | Runs at 25–30 FPS on standard hardware                   |
| 🌍 Cross-platform | Windows, macOS, Linux support                            |

---

## 🖐️ Gesture Map

| Fingers  | Gesture       | Action                                |
| :------: | ------------- | ------------------------------------- |
|   ✊ 0   | Closed Fist   | 🔒 Lock Screen                        |
|   ☝️ 1   | Index Only    | 🖱️ Mouse Mode (move cursor)           |
|   ✌️ 2   | Peace Sign    | 📸 Take Screenshot                    |
|   🤟 3   | Three Fingers | 💻 Open VS Code                       |
|   4️⃣ 4   | Four Fingers  | 🌐 Open Chrome                        |
|   🖐️ 5   | Open Palm     | 🎵 Play / Pause Media                 |
| 🤏 Pinch | Thumb + Index | 👆 Left Mouse Click _(in Mouse Mode)_ |

---

## 🏗️ Architecture

```
gesture_os/
│
├── main.py               ← Entry point & gesture orchestration loop
├── gesture_detector.py   ← MediaPipe integration, finger counting, pinch
├── mouse_controller.py   ← Smooth cursor movement with EMA smoothing
├── screenshot_manager.py ← Timestamped screenshot capture
├── app_launcher.py       ← Cross-platform VS Code / Chrome / lock screen
├── media_controller.py   ← OS media key simulation (play, pause, volume)
├── utils.py              ← Professional UI overlay renderer + FPS counter
│
├── requirements.txt
├── README.md
└── screenshots/          ← Auto-created; gesture screenshots saved here
```

**Design Pattern:** Each feature is a self-contained module with a clean API.  
`main.py` acts as the orchestrator — it reads gesture results and delegates to the right module.  
Adding a new gesture only requires: one entry in `ACTIONS` dict + one `elif` branch.

---

## ⚙️ How It Works

```
Webcam Frame
     ↓
[OpenCV] BGR → RGB conversion
     ↓
[MediaPipe Hands] 21 landmark detection per hand
     ↓
[GestureDetector] Finger counting + pinch distance
     ↓
[main.py] Gesture → Action mapping (with cooldown guard)
     ↓
[MouseController / AppLauncher / ScreenshotManager / MediaController]
     ↓
[utils.UIRenderer] Professional overlay drawn on frame
     ↓
cv2.imshow()  →  You see the result in real time
```

## Smoothing Algorithm:

Mouse movement uses an _Exponential Moving Average_ (EMA) to eliminate hand tremor:

```
smooth_x = α × raw_x  +  (1 − α) × prev_x
```

With `α = 0.18`, approximately 82% of each position comes from history, producing smooth, lag-free control.

---

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- A working webcam
- pip

### Step 1 — Clone the Repository

```bash
git clone https://github.com/JaskiratKourA3-29/GestureOS
cd gesture-os
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — (macOS only) Grant Accessibility Permissions

```
System Settings → Privacy & Security → Accessibility
```

Enable access for Terminal / VS Code / PyCharm (whichever you run the script from).  
PyAutoGUI requires this to control the mouse and keyboard on macOS.

### Step 4 — Run

```bash
python main.py
```

---

## 🎮 Usage

1. Run `python main.py`
2. A window titled **GestureOS** opens showing your webcam feed
3. Hold your hand up in front of the camera
4. Use the gestures in the table above to control your desktop
5. The **HUD panel** on the right shows real-time: finger count, detected gesture, active mode, FPS
6. Press **Q** to quit

**Tips for Best Results:**

- Use in good lighting — avoid backlighting (e.g. bright window behind you)
- Keep hand 40–70 cm from the camera
- Make gestures clearly and hold for ~0.5 s for recognition
- Plain/dark backgrounds improve detection confidence

---

## 🔮 Future Enhancements

The modular architecture is designed for extension. Planned additions:

- [ ] **Multi-gesture sequences** — e.g. swipe left/right for volume
- [ ] **Gesture training** — custom user-defined gestures via TensorFlow
- [ ] **Voice feedback** — TTS confirms actions aloud (e.g. "Screenshot saved")
- [ ] **Productivity Mode** — Open Calculator, Notepad, File Explorer
- [ ] **Presentation Mode** — Next/Previous slide, exit fullscreen
- [ ] **Multi-hand support** — Two-hand gesture combos
- [ ] **Streamlit dashboard** — Web-based control panel & gesture log
- [ ] **Smart Home Mode** — MQTT/IoT device control triggers

---

## 🧠 Learning Outcomes

Building this project demonstrates practical knowledge of:

| Skill                     | Applied As                                       |
| ------------------------- | ------------------------------------------------ |
| **Computer Vision**       | Real-time webcam processing, frame pipeline      |
| **Machine Learning**      | MediaPipe hand landmark model inference          |
| **HCI Design**            | Gesture-to-action UX mapping, cooldown UX        |
| **Desktop Automation**    | PyAutoGUI mouse/keyboard/screenshot control      |
| **Software Architecture** | Modular design, separation of concerns           |
| **Cross-platform Dev**    | Windows / macOS / Linux branch logic             |
| **Real-time Systems**     | FPS optimisation, EMA smoothing, cooldown guards |

---

## 🛠️ Tech Stack

| Technology  | Role                                         |
| ----------- | -------------------------------------------- |
| Python 3.9+ | Core language                                |
| OpenCV      | Webcam capture, frame processing, UI overlay |
| MediaPipe   | AI hand landmark detection (21 keypoints)    |
| PyAutoGUI   | Mouse/keyboard/screenshot automation         |
| NumPy       | Coordinate math, smoothing                   |

---

## 📸 Screenshots


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/8e02abb7-859d-4a38-ac64-34e0f3159ccf" />

> `screenshots/` folder is created automatically.
> Run the app and use the **2-finger gesture** to capture screens!

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with 🖐️ + AI | Computer Vision · HCI · Desktop Automation

_Star ⭐ this repo if you found it useful!_

</div>
