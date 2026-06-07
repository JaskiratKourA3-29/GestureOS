<div align="center">

# 🎨 HandCanvas AI

### Real-Time Air Drawing System

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-FF6F00?style=flat-square&logo=google&logoColor=white)](https://mediapipe.dev/)

_Draw in the air with your finger — no mouse, no pen, no touch required._  
_Your hand IS the paintbrush._

</div>

---

## 🎯 What Is HandCanvas AI?

HandCanvas AI is a **virtual whiteboard powered by real-time hand tracking**.  
Using only a standard webcam, it detects the exact position of your index finger and paints onto an invisible canvas overlaid on your camera feed.

Think of it as **augmented reality drawing** — you draw in mid-air and see the result live on screen.

---

## ✋ Gesture Controls

|     Gesture      | Action                                  |
| :--------------: | --------------------------------------- |
| ☝️ **1 finger**  | Draw — move index tip to paint          |
| ✌️ **2 fingers** | Erase — hover index over drawn area     |
| 🤟 **3 fingers** | Cycle colour (8 colours available)      |
| 4️⃣ **4 fingers** | Cycle brush size (5 sizes)              |
| 🖐️ **5 fingers** | Save canvas as PNG                      |
|   🤏 **Pinch**   | Clear entire canvas                     |
|      **Q**       | Quit (auto-saves if canvas has content) |

---

## 🎨 Colour Palette

`White` · `Green` · `Sky Blue` · `Red` · `Orange` · `Purple` · `Cyan` · `Yellow`

---

## 🚀 Run It

```bash
pip install mediapipe opencv-python numpy
python handcanvas.py
```

---

## 💡 Why This Is Impressive

- **No ML training required** — pure geometric finger tracking
- **Real-time at 25–30 FPS** — smooth stroke rendering
- **Layer blending** — paint layer composited over live webcam using bitwise ops
- **Auto-save on exit** — never lose your work
- **Single-file architecture** — entire system in one clean Python file

---

## 🔮 Extension Ideas

- Shape recognition (circle, line, rectangle detection) using contour analysis
- OCR on drawn text — draw letters, get recognised text output
- Multi-colour strokes based on hand orientation
- Undo/redo stack
- Collaborative drawing over network sockets

---

<div align="center">

_Built with Computer Vision + AI_

</div>
