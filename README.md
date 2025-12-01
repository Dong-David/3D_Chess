# 3D Chess

**3D Chess** is a 3D chess simulation project with a rendering core built using SDL2, Numba, and NumPy. The goal is to create an intuitive, interactive environment that is easily scalable for developing advanced features like AI.

---

## 📌 Key Features

* 3D rendering of the chessboard and pieces.
* Flexible camera control (pan, zoom, rotate view).
* Player interaction handling: piece selection, valid move highlighting.
* Movement logic strictly adhering to standard chess rules.
* Scalable code architecture for adding new features.

---

## 📁 Directory Structure

Below is the full structure and detailed description of each directory. (**Note:** the `__pycache__` directory contains Python cache files; it does not affect the code and can be ignored.)

```text
3D_Chess/
│
├── main.py
│   • Main entry point for the Python version.
│   • Initializes the game, renderer, loads resources, and runs the main loop.
│
├── chess_pieces.py
│   • Definitions for chess pieces, attributes, and model loading data.
│   • Handles piece types, IDs, colors, and mapping to 3D models.
│
├── compiler/
│   • Processor and compiler for Numba JIT functions.
│   • Checks shader errors, supports shader loading and linking.
│
├── renderer/
│   • The entire 3D rendering system.
│   • Camera, lighting, shaders, model drawing, board drawing, view control.
│   • Execution of per-frame render functions.
│
├── loader/
│   • Resource loader for `.glb`, `.pkl` models, and textures.
│   • Converts model data into drawable OpenGL formats.
│
├── histories/
│   • Stores game history.
│   • JSON format: records all moves.
│   • TXT format: summary or simplified history.
│
├── res/
│   ├── chess_pieces.glb
│   │   • 3D model file for all chess pieces (GLB format).
│   │
│   ├── chess_pieces.pkl
│   │   • Pre-processed data for faster loading.
│   │
│   └── sounds/
│       • Game sound assets:
│       • Move.wav – Piece movement sound.
│       • Capture.wav – Piece capture sound.
│       • Check.wav – Check warning sound.
│       • Notify.wav – Notification sound.
│       • Victory.wav – Victory sound.
```

---

## 🚀 Installation & Usage
### 🐍 Python Version

#### Requirements

* Python 3.10+
* pip
* Libraries: numba, numpy, scipy, trimesh, Pillow

#### Installation

```bash
pip install numba numpy scipy trimesh Pillow
```

#### Run

```bash
python3 main.py
```
---

## 🤖 AI Modes

The project supports 4 AI levels:

* **Easy** – Random moves, no calculation.
* **Medium** – Prioritizes capturing pieces when possible.
* **Hard** – Looks ahead 1 move to avoid losing pieces or to gain an advantage.
* **Expert** – Looks ahead 2 moves (Minimax depth 2), defends well, and counter-attacks.

## 🧩 Future Development

* Add AI algorithms (Minimax, Alpha-Beta)
* Piece animations
* Shadow and reflection effects
* Online / LAN mode
