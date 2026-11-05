"""Calibration module: detect the project sheet and warp it to a top-down square.

The physical working area is an 18x18 cm white sheet framed by a thin black
border line with solid black squares at the four corners. Together these form a
single connected dark rectangular loop around the sheet, which is a strong and
lighting-robust feature even on a low-quality webcam. The camera may be rotated
or tilted, so we must correct the perspective before analyzing object positions.

Pipeline:
    1. find_sheet_corners  -> locate the 4 corners of the black frame
    2. order_corners       -> sort corners into a fixed order (TL, TR, BR, BL)
    3. warp_sheet          -> perspective-correct to a fixed-size square image
    4. pixel_to_normalized -> map warped pixel coordinates to (0.0-1.0) with
                              the TOP-LEFT corner as the origin, per the project
                              specification: (0,0) top-left to (1,1) bottom-right
"""

import cv2
import numpy as np


def _quad_squareness(pts):
    """Return the side-length ratio (shortest/longest) of a 4-point polygon.

    A value near 1.0 means the four sides are similar in length, as expected for
    the square sheet frame. This is used to score candidate quads.
    """
    pts = np.asarray(pts, dtype=np.float32)
    sides = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
    return float(min(sides) / (max(sides) + 1e-6))


def _extract_frame_quad(dark_mask, image_area, min_area_ratio):
    """Find the largest dark contour and reduce it to a 4-corner polygon.

    Args:
        dark_mask: Binary image where the dark frame is white (255).
        image_area: Total pixel area of the source image.
        min_area_ratio: Minimum contour area as a fraction of the image.

    Returns:
        A (4, 2) float32 array of corners, or None if no suitable quad is found.
    """
    contours, _ = cv2.findContours(
        dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / image_area
    # Reject tiny blobs (dark objects/shadows) and near-full-frame grabs.
    if area_ratio < min_area_ratio or area_ratio > 0.95:
        return None

    perimeter = cv2.arcLength(largest, True)
    for eps_factor in np.linspace(0.02, 0.08, 7):
        approx = cv2.approxPolyDP(largest, eps_factor * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return None


def find_sheet_corners(img_bgr, dark_threshold=None, min_area_ratio=0.2):
    """Detect the 4 corners of the black frame surrounding the sheet.

    Strategy: the thin black border plus the solid black corner squares form one
    large connected dark region (a rectangular loop). We threshold for dark
    pixels, take the largest dark contour, and reduce it to a 4-point polygon
    with approxPolyDP. Because the frame runs along the whole perimeter, a single
    object covering one spot barely affects the detected quad, and the thick
    corner squares keep the corners well-defined.

    Robustness to lighting: a single fixed threshold fails across bright and dim
    scenes, so we sweep several thresholds, extract a candidate quad at each, and
    keep the one whose sides are most equal in length (most square-like). This
    reliably locks onto the sheet frame regardless of overall brightness.

    Args:
        img_bgr: Input image in BGR color space.
        dark_threshold: Optional fixed grayscale threshold. If None (default),
            a range of thresholds is tried and the best quad is selected.
        min_area_ratio: Minimum frame area as a fraction of the whole image.
            Guards against picking up small dark objects or shadows.

    Returns:
        A (4, 2) float32 array of corner points, or None if no valid quad found.
    """
    img_h, img_w = img_bgr.shape[:2]
    image_area = img_h * img_w

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    # Decide which thresholds to try. A caller-supplied value takes priority;
    # otherwise sweep a range that covers both dim and bright lighting.
    thresholds = (
        [dark_threshold] if dark_threshold is not None
        else list(range(80, 171, 10))
    )

    best_corners = None
    best_score = -1.0
    for threshold in thresholds:
        _, dark_mask = cv2.threshold(
            blurred, threshold, 255, cv2.THRESH_BINARY_INV
        )
        # Close small gaps so the thin border stays connected into one loop.
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

        quad = _extract_frame_quad(dark_mask, image_area, min_area_ratio)
        if quad is None:
            continue

        # Prefer the most square-like candidate (sides of equal length).
        score = _quad_squareness(quad)
        if score > best_score:
            best_score = score
            best_corners = quad

    return best_corners


def order_corners(pts):
    """Order 4 points consistently as top-left, top-right, bottom-right, bottom-left.

    Uses the classic sum/difference trick:
        - top-left has the smallest (x + y)
        - bottom-right has the largest (x + y)
        - top-right has the smallest (y - x)
        - bottom-left has the largest (y - x)

    Args:
        pts: A (4, 2) array of corner points in any order.

    Returns:
        A (4, 2) float32 array ordered [TL, TR, BR, BL].
    """
    pts = np.asarray(pts, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1).ravel()  # y - x
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left

    return ordered


def warp_sheet(img_bgr, corners, size=800):
    """Perspective-correct the sheet to a fixed-size top-down square.

    Args:
        img_bgr: Input image in BGR color space.
        corners: A (4, 2) array of the sheet corners (any order).
        size: Side length in pixels of the output square.

    Returns:
        A tuple (warped_img, transform_matrix). The matrix can later be used
        to map points between original and warped coordinates.
    """
    src = order_corners(corners)

    # Destination is a perfect square. Order matches order_corners: TL, TR, BR, BL.
    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_bgr, transform, (size, size))
    return warped, transform


def pixel_to_normalized(x, y, size):
    """Map a warped pixel coordinate to normalized (0.0-1.0) sheet coordinates.

    Per the project specification, the origin (0, 0) is the TOP-LEFT corner and
    (1, 1) is the bottom-right corner. This matches the warped image's own pixel
    layout (rows grow downward), so no axis flip is required.

    Args:
        x: Pixel x within the warped image.
        y: Pixel y within the warped image.
        size: Side length of the warped square in pixels.

    Returns:
        A (norm_x, norm_y) tuple of floats in the range [0.0, 1.0].
    """
    norm_x = x / size
    norm_y = y / size  # top-left origin: y grows downward, same as image rows
    return norm_x, norm_y
