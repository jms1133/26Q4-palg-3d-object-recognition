# Offline Object Detection (Shape, Color, Position)

A classical computer-vision application that processes a live webcam stream to
identify the **shape**, **color**, and **normalized coordinates** of
single-colored objects placed on an 18 cm x 18 cm workspace. It runs completely
offline using OpenCV and NumPy, with no network access and no pre-trained deep
learning models.

## Requirements

- Python 3.9+ (developed on Python 3.13)
- A webcam
- The packages listed in `requirements.txt` (OpenCV and NumPy)

## Setup

From the project root, create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead.

## Running

Run from the project root so the `src` package resolves:

```bash
python main.py
```

A window titled **Canvas** opens showing the warped top-down view of the
workspace with detected objects outlined and labeled.

### Controls

| Key | Action                                   |
|-----|------------------------------------------|
| `q` | Quit                                     |
| `r` | Force re-analysis of the current frame   |
| `c` | Save the current canvas to a PNG         |

Captured images are written to `images/main_test/` (the folder is created
automatically on the first capture).

## Camera selection

The capture device is chosen by `CAMERA_INDEX` near the top of `main.py`
(default `0`). This index is **machine-dependent**: the same USB webcam can
enumerate as `0` on one computer and `1` or `2` on another, and it is not always
distinct from a built-in camera. If the wrong device opens, or the camera fails
to open, edit `CAMERA_INDEX` and try `0`, `1`, or `2`.

## Notes for running on Linux

- The application needs a graphical session; `cv2.imshow` cannot open a window
  over a headless/SSH connection.
- Camera exposure and white balance are handled automatically in software, so no
  camera-specific driver settings are required.

## Hardware and physical setup

- **USB or built-in webcam**, positioned for a near bird's-eye (zenital) view of
  the canvas. The reference setup rests the webcam on the laptop lid facing
  outward, opens the lid to about 90 degrees, and tilts the webcam down toward
  the canvas. A mild perspective tilt is fine; calibration corrects for it.
  Arrange the camera so it does not cast its own shadow onto the white canvas.
- **Matte white 18 cm x 18 cm canvas** (cardboard sheet). This is the measured
  surface: it maps to the normalized unit square from `(0.0, 0.0)` (top-left) to
  `(1.0, 1.0)` (bottom-right).
- **Dark / black mat underneath** the canvas. Calibration works by finding the
  white canvas as the brightest region against this dark background, so the
  strong white-on-black contrast is required. No corner markers are needed.
- **Even, neutral-colored lighting**, bright enough that the white canvas is
  clearly the brightest area in the frame and objects are not lost in shadow.
  Avoid any part of the setup casting shadows across the canvas.
- **Single-colored (monochromatic) objects** placed on the canvas. Objects may
  touch side by side; stacking objects on top of each other is not supported.

## Project structure

```
main.py             Live video loop that ties the pipeline together
requirements.txt    Runtime dependencies
src/
  calibration.py    Corner detection and perspective warp
  color.py          HSV color masking and white balance
  hsv_sampler.py    Interactive HSV sampling helper
  segmentation.py   Touching-object separation (distance transform / watershed)
  shapes.py         Shape classification and object description
  stability.py      Scene-change / motion detection
```
