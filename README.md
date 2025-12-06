# 3D Chess

**3D Chess** is a high-performance 3D chess simulation built from scratch using Python. The project leverages **SDL2** for rendering and **Numba** for accelerating computational logic. The goal is to create a visually intuitive, interactive environment capable of running advanced AI algorithms without compromising frame rates.

---

<img width="824" height="650" alt="Screenshot 2025-12-06 at 21 29 42" src="https://github.com/user-attachments/assets/80302d4e-ebf0-483b-b621-c329470830f9" />

---

## 📌 Key Features

* **High-Performance Rendering:** Custom 3D engine using SDL2 and NumPy.
* **Advanced AI Engine:** Numba-accelerated Minimax algorithm with Alpha-Beta Pruning and Quiescence Search.
* **Multithreading:** AI calculations run on a separate daemon thread to ensure the UI remains responsive (no freezing during "thinking" time).
* **Interactive Camera:** Flexible control (pan, zoom, rotate) to view the board from any angle.
* **Game Logic:** Strict adherence to standard chess rules, including Castling, En Passant, and Promotion.
* **Scalable Architecture:** Modular design separating rendering, game logic, and AI computation.

---

## 📁 Directory Structure

Below is the structure of the project source code.

```text
3D_Chess/
│
├── main.py
│   • Entry point. Initializes the game loop, renderer, and handles thread management.
│   • Bridges the UI events with the AI logic.
│
├── chess_ai.py
│   • The "brain" of the computer.
│   • Contains the ChessAI class, Numba-compiled Minimax engine, and evaluation functions (PST).
│
├── chess_pieces.py
│   • Definitions for chess pieces, attributes, and model mapping.
│
├── compiler/
│   • Numba JIT compiler configurations and shader management.
│
├── renderer/
│   • Core 3D rendering system (Camera, Lighting, Shaders, Draw calls).
│
├── loader/
│   • Resource loader for .glb models and textures.
│
├── histories/
│   • Stores game logs in JSON and TXT formats.
│
├── res/
│   ├── chess_pieces.glb   • 3D models for pieces.
│   ├── chess_pieces.pkl   • Optimized cache for faster loading.
│   └── sounds/            • Audio assets (Move, Capture, Check, Victory).

```

---

## 🚀 Installation & Usage

### 🐍 Python Version

#### Requirements

* Python 3.10+
* **Libraries:** `numba`, `numpy`, `scipy`, `trimesh`, `Pillow`, `pysdl2`

#### Installation

```bash
pip install numba numpy scipy trimesh Pillow pysdl2 pysdl2-dll
```

#### Run

```bash
python3 main.py
```

---
