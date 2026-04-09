# NeuralGaze Engine

**NeuralGaze Engine** is a professional-grade, modular computer vision framework designed for hands-free human-computer interaction. It utilizes advanced facial landmark estimation and biometric tracking to provide high-precision mouse control and intent-based gesture recognition.

![Project Status](https://img.shields.io/badge/Status-Production--Ready-green)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Computer Vision](https://img.shields.io/badge/Technology-MediaPipe%20%2F%20OpenCV-orange)

## 🚀 Key Features

- **Modular Engineering Architecture**: Decoupled systems for Vision Processing, Mathematical Projection, and Hardware Interaction, ensuring scalability and maintainability.
- **Biometric Blink-to-Click**: Proprietary time-based blink detection algorithm distinguishing between natural blinks, intentional Left Clicks (short), and Right Clicks (long).
- **Kinematic Smoothing Engine**: Advanced noise reduction and dead-zone logic providing steady and fluid cursor movements even with standard webcams.
- **Hybrid Pose-Face Estimation**: Combines facial axis direction with body pose landmarks for increased stability and head-tilt compensation.
- **Professional Analytics HUD**: Real-time Heads-Up Display showing signal quality, FPS, and tracking metrics.

## 🛠️ Technical Overview

The engine is structured into specialized modules to adhere to the Single Responsibility Principle (SRP):
- `modules/vision_engine.py`: Core AI pipeline and frame acquisition.
- `modules/mouse_engine.py`: Hardware abstraction layer and cursor kinematics.
- `utils/math_utils.py`: Pure computational logic for gaze projection and biometric metrics.
- `utils/display_hud.py`: Rendering engine for the professional analysis interface.

## 📥 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Dourayed-Smari/NeuralGaze-Engine.git
   cd NeuralGaze-Engine
   ```

2. **Setup virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

Launch the engine:
```bash
python src/app.py
```

### Controls:
- **[C]** - Calibrate: Look at the center crosshair and press C to align the optical axis.
- **[M]** - Toggle Mode: Switch between `DYNAMIC KINEMATICS` (Fast) and `EYE-ONLY` (Slow).
- **[L]** - Toggle HUD: Show/Hide the AI facial mesh.
- **[Blink]** - Interaction:
  - Short Blink (0.2s - 1.0s) -> **Left Click**
  - Long Blink (1.0s - 3.0s) -> **Right Click**
- **[Q]** - Secure Shutdown.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---
*Developed as a demonstration of advanced Computer Vision and Software Engineering principles.*
