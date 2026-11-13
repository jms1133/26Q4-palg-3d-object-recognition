"""HSV sampling / verification helper (offline tuning tool).

This is a developer utility, not part of the runtime pipeline. It lets us click
inside an object in a captured frame and read back the HSV statistics of that
object's clean interior, together with the label that the production classifier
(``color.classify_color_hsv``) assigns to the same sample. That makes it easy to
confirm that the measured thresholds in ``color.py`` are still correct for the
current object set and lighting, or to spot where they drift.

To keep the measurement faithful to production, the frame is first run through
the real calibration warp (``calibration.find_sheet_corners`` +
``calibration.warp_sheet``) so sampling happens on the same flat canvas the
detector sees. If no canvas is found, the raw frame is used instead.

Each click samples a small square patch, erodes it to an interior core (per the
project's interior-sampling guidance: avoid glossy tops and shadowed edges), and
reports the median plus min/max of H, S and V across that core.

Usage:
    python -m src.hsv_sampler [IMAGE_PATH]

If IMAGE_PATH is omitted, the newest PNG in images/camera_test is used.

Controls:
    left click : sample the clicked point
    s          : save an annotated copy next to the source image
    q / ESC    : quit
"""

import glob
import os
import sys

import cv2
import numpy as np

from . import calibration
from . import color

# Default location to search when no explicit image path is given.
DEFAULT_CAPTURE_DIR = "images/camera_test"

# Half-size (in pixels) of the square patch sampled around a click. The full
# patch is (2 * PATCH_HALF + 1) on a side.
PATCH_HALF = 12

# Erosion kernel used to shrink the patch to an interior core, dropping edge and
# highlight pixels so the sampled color reflects the object body (per CLAUDE.md
# interior-sampling guidance).
_INTERIOR_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def newest_capture(capture_dir=DEFAULT_CAPTURE_DIR):
    """Return the path to the most recently modified PNG in a directory.

    Args:
        capture_dir: Directory to search for capture PNGs.

    Returns:
        The newest PNG path, or None if the directory has no PNGs.
    """
    candidates = glob.glob(os.path.join(capture_dir, "*.png"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def warp_if_possible(img_bgr):
    """Warp the frame to the top-down canvas if the canvas can be found.

    Falls back to the raw frame so the tool still works on images where the
    canvas is cropped or undetected.

    Args:
        img_bgr: Input image in BGR color space.

    Returns:
        A tuple (image_to_sample, warped) where warped is True if the
        calibration warp was applied.
    """
    corners = calibration.find_sheet_corners(img_bgr)
    if corners is None:
        return img_bgr, False
    warped, _ = calibration.warp_sheet(img_bgr, corners)
    return warped, True


def sample_patch(img_bgr, x, y, half=PATCH_HALF):
    """Sample an eroded interior patch around (x, y) and return HSV statistics.

    A square patch is cut around the click and eroded to an interior core so that
    object edges, shadowed borders and specular highlights contribute less. The
    core's HSV median and range summarize the local color.

    Args:
        img_bgr: Image in BGR color space to sample from.
        x: Click x coordinate in image pixels.
        y: Click y coordinate in image pixels.
        half: Half-size of the square patch in pixels.

    Returns:
        A dict with keys 'median', 'low', 'high' (each an (H, S, V) int tuple)
        and 'count' (number of interior pixels sampled), or None if the patch
        would fall outside the image or erodes to nothing.
    """
    height, width = img_bgr.shape[:2]
    x0, x1 = x - half, x + half + 1
    y0, y1 = y - half, y + half + 1
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        return None

    patch = img_bgr[y0:y1, x0:x1]

    # Erode a full-ones mask to keep only the patch's interior core.
    core = np.ones(patch.shape[:2], dtype=np.uint8) * 255
    core = cv2.erode(core, _INTERIOR_KERNEL)
    if cv2.countNonZero(core) == 0:
        return None

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    pixels = hsv[core > 0]

    median = np.median(pixels, axis=0).astype(int)
    low = pixels.min(axis=0).astype(int)
    high = pixels.max(axis=0).astype(int)
    return {
        "median": tuple(int(c) for c in median),
        "low": tuple(int(c) for c in low),
        "high": tuple(int(c) for c in high),
        "count": int(pixels.shape[0]),
    }


def format_sample(stats):
    """Build a one-line report string for a sampled patch.

    Args:
        stats: The dict returned by sample_patch.

    Returns:
        A human-readable string including the classifier's predicted label.
    """
    median = stats["median"]
    label = color.classify_color_hsv(median)
    return (
        f"label={label:<8} "
        f"median HSV={median} "
        f"range H[{stats['low'][0]}-{stats['high'][0]}] "
        f"S[{stats['low'][1]}-{stats['high'][1]}] "
        f"V[{stats['low'][2]}-{stats['high'][2]}] "
        f"(n={stats['count']})"
    )


def _on_mouse(event, x, y, flags, userdata):
    """Mouse callback: sample on left click and record an annotation."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    image = userdata["image"]
    stats = sample_patch(image, x, y)
    if stats is None:
        print("Click too close to the border or patch empty; try again.")
        return
    line = format_sample(stats)
    print(line)
    userdata["marks"].append((x, y, color.classify_color_hsv(stats["median"])))


def _draw_marks(image, marks):
    """Return a copy of image with sampled points annotated."""
    annotated = image.copy()
    for x, y, label in marks:
        cv2.circle(annotated, (x, y), PATCH_HALF, (0, 0, 0), 1)
        cv2.circle(annotated, (x, y), 2, (0, 0, 255), -1)
        cv2.putText(
            annotated, label, (x + PATCH_HALF, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA
        )
        cv2.putText(
            annotated, label, (x + PATCH_HALF, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
    return annotated


def run(image_path):
    """Open the interactive sampler window for one image.

    Args:
        image_path: Path to the capture PNG to sample.

    Returns:
        0 on normal exit, 1 if the image could not be read.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: could not read image '{image_path}'.")
        return 1

    sample_image, warped = warp_if_possible(image)
    print(f"Loaded {image_path} (warped={warped}).")
    print("Left-click inside an object to sample. Keys: s=save, q/ESC=quit.")

    window = "HSV sampler"
    userdata = {"image": sample_image, "marks": []}
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, _on_mouse, userdata)

    while True:
        cv2.imshow(window, _draw_marks(sample_image, userdata["marks"]))
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            root, ext = os.path.splitext(image_path)
            out_path = f"{root}_hsv_marked{ext}"
            cv2.imwrite(out_path, _draw_marks(sample_image, userdata["marks"]))
            print(f"Saved annotated copy -> {out_path}")

    cv2.destroyAllWindows()
    return 0


def main(argv):
    """CLI entry point: resolve the image path and run the sampler.

    Args:
        argv: Argument list excluding the program name.

    Returns:
        Process exit code.
    """
    if argv:
        image_path = argv[0]
    else:
        image_path = newest_capture()
        if image_path is None:
            print(f"Error: no PNG captures found in '{DEFAULT_CAPTURE_DIR}'.")
            return 1

    return run(image_path)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
