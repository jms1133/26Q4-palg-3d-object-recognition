"""Main application: live object detection over the warped canvas.

This ties the modules into one video loop, following the project pipeline:

    1. Grab a webcam frame.
    2. Warp it to the top-down unit-square canvas (calibration module).
    3. Feed the frame to the stability detector. Classification runs only when
       the scene has settled, so labels are computed once per arrangement instead
       of flickering while a hand places or moves an object.
    4. On settling: for each target color, build its HSV mask, split touching
       same-color objects, and describe every object (shape, color, normalized
       position). The results are cached.
    5. Every frame: draw the cached detections (contour outline, label, centroid,
       and normalized coordinates) onto the warped view and show it.

Controls:
    q - quit
    r - force re-analysis of the current frame
    c - save the current canvas to a timestamped PNG

Run from the project root so the ``src`` package resolves:

    python main.py
"""

import os
from datetime import datetime

import cv2
import numpy as np

from src import calibration, color, segmentation, shapes
from src.stability import StabilityDetector

# Index of the capture device. This is machine-dependent: the same USB webcam
# can enumerate as 0 on one computer and 1 or 2 on another, and it is not always
# distinct from any built-in camera. If the wrong device opens, try 0, 1, or 2.
CAMERA_INDEX = 0

# Side length of the warped canvas in pixels. Must match the size passed to
# describe_object so normalized coordinates are correct.
WARP_SIZE = 800

# Directory where captured canvas snapshots are written.
CAPTURE_DIR = "images/main_test"

# Black margin in pixels drawn around the warped canvas for DISPLAY ONLY. This is
# a purely cosmetic frame so the canvas is not cropped flush to the window edge;
# detection, coordinates and the warp itself are unaffected.
DISPLAY_MARGIN = 60

# Target color buckets to detect, drawn from color.COLOR_RANGES.
TARGET_COLORS = ("red", "yellow", "green", "blue")

# BGR outline/label color per detected object color, for readable overlays.
DRAW_COLORS = {
    "red": (0, 0, 255),
    "yellow": (0, 215, 255),
    "green": (0, 180, 0),
    "blue": (255, 100, 0),
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def detect_objects(warped_bgr):
    """Run the full per-color detection pipeline on one warped frame.

    Args:
        warped_bgr: The warped top-down canvas in BGR color space.

    Returns:
        A list of (contour, description) pairs, where description is the dict
        returned by shapes.describe_object.
    """
    # Neutralize the lighting color cast using the matte white sheet as a neutral
    # reference, so warm/cold lighting does not shift objects out of their tuned
    # HSV ranges. The sheet fills the warped canvas, giving a reliable reference.
    white_patch = color.sample_white_patch(warped_bgr)
    balanced = color.white_balance(warped_bgr, white_patch)
    hsv = cv2.cvtColor(balanced, cv2.COLOR_BGR2HSV)
    detections = []
    for color_name in TARGET_COLORS:
        mask = color.build_color_mask(hsv, color_name)
        for contour in segmentation.segment_color_mask(mask):
            description = shapes.describe_object(contour, hsv, WARP_SIZE)
            if description is not None:
                detections.append((contour, description))
    return detections


def draw_detections(canvas, detections):
    """Overlay contours, labels, centroids and coordinates onto the canvas.

    Args:
        canvas: The warped BGR image to draw on (modified in place).
        detections: The list of (contour, description) pairs to render.
    """
    for contour, info in detections:
        draw_color = DRAW_COLORS.get(info["color"], (255, 255, 255))
        cv2.drawContours(canvas, [contour], -1, draw_color, 2)

        centroid = info["centroid"]
        if centroid is None:
            continue
        cx, cy = centroid
        cv2.circle(canvas, (cx, cy), 4, draw_color, -1)

        norm_x, norm_y = info["position"]
        label = f"{info['color']} {info['shape']}"
        coords = f"({norm_x:.2f}, {norm_y:.2f})"

        # Draw the two text lines stacked above the centroid in plain black. A
        # single solid color keeps the labels easy to read and avoids the
        # distracting colored text layer.
        lines = (label, coords)
        for i, text in enumerate(lines):
            y = cy - 10 - (len(lines) - 1 - i) * 18
            cv2.putText(canvas, text, (cx + 8, y), _FONT, 0.5, (0, 0, 0), 1,
                        cv2.LINE_AA)


def draw_status(canvas, state, count):
    """Draw a small status banner with motion/stability and object count.

    Args:
        canvas: The warped BGR image to draw on (modified in place).
        state: The latest StabilityState.
        count: Number of currently reported objects.
    """
    status = "STABLE" if state.is_stable else "MOVING"
    banner = f"{status} | motion={state.motion:.3f} | objects={count}"
    cv2.putText(canvas, banner, (10, 24), _FONT, 0.6, (0, 0, 0), 1, cv2.LINE_AA)


def frame_on_black(canvas, margin=DISPLAY_MARGIN):
    """Center the overlaid canvas on a larger solid-black background.

    This is a display-only presentation step: it adds a uniform black border so
    the canvas is not cropped flush to the window edge. It does not alter the warp
    or any coordinates, which are computed on the original WARP_SIZE canvas before
    this is called.

    Args:
        canvas: The warped BGR image (already overlaid) to frame.
        margin: Black border width in pixels added on every side.

    Returns:
        A new BGR image of size (h + 2*margin, w + 2*margin) with the canvas
        centered on black.
    """
    h, w = canvas.shape[:2]
    framed = np.zeros((h + 2 * margin, w + 2 * margin, 3), dtype=canvas.dtype)
    framed[margin:margin + h, margin:margin + w] = canvas
    return framed


def save_capture(canvas):
    """Write the current canvas to a timestamped PNG in CAPTURE_DIR.

    Args:
        canvas: The warped BGR image (with overlays) to save.

    Returns:
        The path the image was written to.
    """
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(CAPTURE_DIR, f"capture_{timestamp}.png")
    cv2.imwrite(path, canvas)
    return path


def main():
    """Open the camera and run the live detection loop."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: cannot open camera at index {CAMERA_INDEX}.")
        return

    detector = StabilityDetector()
    prev_corners = None
    detections = []

    print(
        "Running. Press 'q' to quit, 'r' to force re-analysis, 'c' to save a "
        "capture, 'f' to peek at the full camera frame."
    )
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Error: cannot read frame from camera.")
            break

        # Hold the last good warp through a momentary corner-detection miss.
        corners = calibration.find_sheet_corners(frame, prev_corners=prev_corners)
        if corners is None:
            cv2.imshow("Canvas", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
            continue
        prev_corners = corners
        warped, _ = calibration.warp_sheet(frame, corners, size=WARP_SIZE)

        # Classify only when the scene has just settled, so labels are stable.
        state = detector.update(frame)
        if state.just_settled:
            detections = detect_objects(warped)

        canvas = warped.copy()
        draw_detections(canvas, detections)
        draw_status(canvas, state, len(detections))

        # Present the overlaid canvas inside a cosmetic black frame so it is not
        # cropped flush to the window edge. Detection and coordinates were already
        # computed on `warped` above, so this display step changes nothing.
        display = frame_on_black(canvas)
        cv2.imshow("Canvas", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            detections = detect_objects(warped)
        if key == ord("c"):
            path = save_capture(display)
            print(f"Capture: {path}")
        if key == ord("f"):
            # Briefly show the full raw camera frame so the operator can re-frame
            # the canvas and background. This is a live-setup aid only; it does not
            # affect detection. The 1000 ms wait blocks the loop on purpose.
            cv2.imshow("Full frame", frame)
            cv2.waitKey(1000)
            cv2.destroyWindow("Full frame")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
