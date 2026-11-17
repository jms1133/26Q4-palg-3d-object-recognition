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

| Key | Action                                                     |
|-----|------------------------------------------------------------|
| `q` | Quit                                                       |
| `r` | Force re-analysis of the current frame                     |
| `c` | Save the current canvas to a PNG                           |
| `f` | Briefly show the full raw camera frame (to re-frame setup) |
| `n` | Decrease camera exposure (only when exposure control is active) |
| `m` | Increase camera exposure (only when exposure control is active) |

Captured images are written to `images/main_test/` (the folder is created
automatically on the first capture).

## Camera selection

The capture device is chosen by `CAMERA_INDEX` near the top of `main.py`
(default `0`). This index is **machine-dependent**: the same USB webcam can
enumerate as `0` on one computer and `1` or `2` on another, and it is not always
distinct from a built-in camera. If the wrong device opens, or the camera fails
to open, edit `CAMERA_INDEX` and try `0`, `1`, or `2`.

## Camera exposure control (optional)

A dark background can make the camera's auto-exposure overexpose and wash out the
white canvas. The application can optionally lock the exposure so the canvas
highlights sit at a fixed level, keeping colors saturated. This is **best-effort**
and platform-dependent; if it is not available the app silently falls back to the
camera's auto-exposure and runs normally.

- **Linux:** handled automatically through OpenCV's V4L2 properties
  (`CAP_PROP_AUTO_EXPOSURE` / `CAP_PROP_EXPOSURE`). No extra setup is needed. If
  the specific camera/driver rejects the write, auto-exposure is used instead.
- **macOS (optional):** macOS/AVFoundation ignores OpenCV exposure writes, so a
  small native helper binary named `uvc-util-app` is used if present in the
  project root. It is **not** required and is not included in the submission. To
  enable it, run these three commands **from the project root**, one at a time:

  ```bash
  git clone https://github.com/jtfrey/uvc-util.git
  ```

  ```bash
  (cd uvc-util/src && gcc -o "$OLDPWD/uvc-util-app" -framework IOKit -framework Foundation uvc-util.m UVCController.m UVCType.m UVCValue.m)
  ```

  ```bash
  rm -rf uvc-util
  ```

  Each build command is a single line (no line-continuation backslashes). The
  second line compiles the binary into the project root; the third removes the
  now-unneeded source. If the binary is absent, macOS simply runs with
  auto-exposure. Exposure units are camera-specific; the
  built-in values target the Aukey webcam used during development.

When exposure control is active, the `n` / `m` keys fine-tune the exposure live.

## Notes for running on Linux

- The application needs a graphical session; `cv2.imshow` cannot open a window
  over a headless/SSH connection.
- White balance is handled automatically in software. Exposure is handled as
  described in the section above and never requires manual driver configuration.

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
  exposure.py       Optional per-platform camera exposure control (macOS/Linux)
  hsv_sampler.py    Interactive HSV sampling helper
  segmentation.py   Touching-object separation (distance transform / watershed)
  shapes.py         Shape classification and object description
  stability.py      Scene-change / motion detection
```
