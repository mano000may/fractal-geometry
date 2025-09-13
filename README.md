# Fractal Explorer in Python

A high-performance, interactive fractal explorer built with Python. This application uses Taichi for GPU acceleration (if available) for real-time exploration and falls back to a highly optimized NumPy renderer. It supports deep zooms using arbitrary-precision math via `mpmath`.

## Features

- **Multiple Renderers:**
  - **Taichi (GPU):** Blazing-fast real-time rendering using `f32` or `f64` precision.
  - **NumPy (CPU):** Vectorized CPU fallback for systems without a compatible GPU.
  - **MPMATH (CPU):** Arbitrary-precision renderer for virtually infinite deep zooms.
- **Interactive Navigation:**
  - Real-time "infinite scroll" panning by dragging the mouse.
  - Zoom with the mouse wheel, centered on the cursor.
  - Click to re-center the view.
- **Multiple Fractal Sets:** Explore a variety of classic and modern fractals, including:
  - Mandelbrot & Julia
  - Burning Ship & Tricorn
  - Mandelbar
  - Newton's Fractal & the highly configurable Nova Fractal
- **Dynamic UI:**
  - A clean sidebar with all controls and an event log.
  - Contextual sliders appear only for the relevant fractal sets (e.g., Julia and Nova constants).
  - Fullscreen mode (F11).
- **Customization:**
  - Adjustable iteration count, power, and color palettes.
  - Automatic iteration scaling for optimal detail at any zoom level.
  - Supersampling (Anti-Aliasing) for high-quality final renders.

## Controls

| Action                  | Control                         |
| ----------------------- | ------------------------------- |
| **Navigation**          |                                 |
| Zoom                    | Mouse Wheel Up/Down             |
| Pan                     | Click and Drag                  |
| Recenter                | Left Click                      |
| **Fractal Selection**   |                                 |
| Cycle Fractal Set       | `F` key                         |
| Cycle Color Palette     | `C` key                         |
| **View & Render**       |                                 |
| Toggle Fullscreen       | `F11` key                       |
| Reset View              | `R` key                         |
| Toggle Auto-Iterations  | `A` key                         |
| Toggle High-Precision   | `P` key                         |
| Adjust Precision        | `[` and `]` keys                |
| Save Screenshot         | `S` key                         |

## Requirements

- Python 3.7+
- `pygame`
- `numpy`
- `taichi` (for GPU acceleration)
- `mpmath` (for high-precision rendering)

## Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/FractalExplorer.git
    cd FractalExplorer
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    python fractal_explorer.py
    ```
