"""Debug capture tool: save the CLEAN warped frame alongside the annotated one.

main.py's capture only saves the canvas with contours and labels already drawn
on it, which corrupts the color masks if that image is later re-run through the
detection pipeline (the drawn outlines cut the blobs). For debugging detection
issues (over-segmentation, occluded objects, threshold tuning) we need the raw
warped canvas with NO overlays, so the pipeline can be re-measured faithfully
offline.

This script mirrors the main.py live loop but, on capture, writes TWO files per
shot into images/debug_exports/:
    <timestamp>_clean.png    the warped canvas, no overlays (for re-analysis)
    <timestamp>_overlay.png  the same frame with detections drawn (for reference)

Controls:
    q - quit
    r - force re-analysis of the current frame
    c - save a clean + overlay capture pair

Run from the project root so the ``src`` package resolves:

    python debug_capture.py
"""

import os
from datetime import datetime

import cv2

from src import calibration
from src.stability import StabilityDetector

# Reuse the exact detection and drawing logic from the live app so captures
# reflect what main.py actually produces.
from main import (
    CAMERA_INDEX,
    WARP_SIZE,
    detect_objects,
    draw_detections,
    draw_status,
)

# Dedicated folder for clean debug exports, kept separate from the ad-hoc
# images/main_test snapshots. Created on first capture.
EXPORT_DIR = "images/debug_exports"


def save_debug_pair(clean_warped, overlay_canvas):
    """Write the clean warped frame and its annotated copy as a timestamped pair.

    Args:
        clean_warped: The warped BGR canvas with NO overlays drawn.
        overlay_canvas: The same canvas with detection overlays drawn.

    Returns:
        A (clean_path, overlay_path) tuple of the written file paths.
    """
    os.makedirs(EXPORT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_path = os.path.join(EXPORT_DIR, f"{timestamp}_clean.png")
    overlay_path = os.path.join(EXPORT_DIR, f"{timestamp}_overlay.png")
    cv2.imwrite(clean_path, clean_warped)
    cv2.imwrite(overlay_path, overlay_canvas)
    return clean_path, overlay_path


def main():
    """Open the camera and run the live loop with clean + overlay capture."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: cannot open camera at index {CAMERA_INDEX}.")
        return

    detector = StabilityDetector()
    prev_corners = None
    detections = []

    print("Debug capture. Press 'q' to quit, 'r' to re-analyze, 'c' to save a pair.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Error: cannot read frame from camera.")
            break

        corners = calibration.find_sheet_corners(frame, prev_corners=prev_corners)
        if corners is None:
            cv2.imshow("Canvas", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
            continue
        prev_corners = corners
        warped, _ = calibration.warp_sheet(frame, corners, size=WARP_SIZE)

        state = detector.update(frame)
        if state.just_settled:
            detections = detect_objects(warped)

        # Keep the clean warp untouched; draw overlays only on a copy.
        clean_warped = warped.copy()
        canvas = warped.copy()
        draw_detections(canvas, detections)
        draw_status(canvas, state, len(detections))
        cv2.imshow("Canvas", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            detections = detect_objects(warped)
        if key == ord("c"):
            clean_path, overlay_path = save_debug_pair(clean_warped, canvas)
            print(f"Saved pair:\n  clean   -> {clean_path}\n  overlay -> {overlay_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
