"""Calibration module: detect the white canvas and warp it to a top-down square.

The physical working area is an 18x18 cm matte WHITE CANVAS (the "wc") resting on
a larger black paper mat, which in turn sits on a light table. The strong white-
on-black contrast makes the canvas the dominant bright region in the frame, so it
is easy to isolate regardless of lighting. White L-brackets are printed on the
mat just outside the canvas corners as a visual aid, but detection relies on the
canvas itself, not on the brackets.

The canvas is the measured surface: it maps directly to the normalized unit
square, from (0.0, 0.0) at the top-left to (1.0, 1.0) at the bottom-right.

Pipeline:
    1. find_sheet_corners  -> locate the 4 corners of the white canvas
    2. order_corners       -> sort corners into a fixed order (TL, TR, BR, BL)
    3. warp_sheet          -> perspective-correct to a fixed-size square image
    4. pixel_to_normalized -> map warped pixel coordinates to (0.0-1.0) with the
                              TOP-LEFT corner as the origin, per the project
                              specification: (0,0) top-left to (1,1) bottom-right

Detection strategy (see find_sheet_corners for detail):
    - Primary: the white canvas is the largest bright blob on the dark mat. Otsu
      splits it cleanly; a 4-point perspective quad is fit to its contour. This is
      accurate whenever the canvas edges are unobstructed.
    - Dent repair: an object overhanging a canvas edge pushes one detected corner
      inward, making the quad non-parallelogram. The most-skewed corner is
      detected and reconstructed from the other three under the assumption that
      the canvas images as a near-parallelogram, so a single occluded corner no
      longer distorts the warp.
    - Temporal lock: the caller may pass the previous good quad so a momentarily
      failed frame reuses the last known-good warp during a live video loop.
"""

import cv2
import numpy as np


def _white_mask(blurred):
    """Threshold the bright canvas and brackets out of the dark mat via Otsu.

    The white canvas and white L-brackets are the dominant bright regions against
    the black mat, so Otsu picks a data-driven split that is robust to overall
    exposure. A morphological open removes speckle, and a close fills small gaps
    (e.g. where a dark object sits on the canvas) so the canvas stays one blob.

    Args:
        blurred: Grayscale, blurred source image.

    Returns:
        A single-channel uint8 mask (0 or 255) of the bright regions.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    _, mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _contour_to_quad(contour):
    """Reduce a contour to a 4-corner perspective quad.

    The canvas appears as a trapezoid under the tilted webcam, so a perspective
    quad (four free corners) fits it better than an axis-aligned box. We sweep a
    range of approxPolyDP tolerances and accept the first that yields exactly four
    vertices. If none does (e.g. the contour is dented by an overhanging object),
    we fall back to the minimum-area rotated rectangle, which ignores small local
    dents at the cost of not following perspective exactly.

    Args:
        contour: An OpenCV contour (Nx1x2 int array).

    Returns:
        A (4, 2) float32 array of corner points.
    """
    perimeter = cv2.arcLength(contour, True)
    for eps_factor in np.linspace(0.02, 0.08, 7):
        approx = cv2.approxPolyDP(contour, eps_factor * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)


def _find_canvas_quad(mask, image_area, min_area_ratio):
    """Find the white canvas as the largest bright blob and reduce it to a quad.

    Args:
        mask: Binary mask of bright regions (canvas + brackets + bright objects).
        image_area: Total pixel area of the source image.
        min_area_ratio: Minimum canvas area as a fraction of the image. Guards
            against locking onto a small bright object when the canvas is hidden.

    Returns:
        A (4, 2) float32 array of canvas corners (unordered), or None.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    canvas = max(contours, key=cv2.contourArea)
    if cv2.contourArea(canvas) / image_area < min_area_ratio:
        return None

    return _contour_to_quad(canvas)


def _corner_angle_errors(ordered):
    """Return each corner's deviation from a right angle, in degrees.

    A canvas viewed under mild perspective images as a near-parallelogram, whose
    interior angles stay close to 90 degrees. When an object overhangs an edge it
    pulls one detected corner inward, collapsing that corner's angle well away
    from 90. Measuring per-corner angle error therefore pinpoints the dented
    corner.

    Args:
        ordered: (4, 2) float32 corners in [TL, TR, BR, BL] order.

    Returns:
        A length-4 float array of absolute angle errors (degrees) per corner.
    """
    errors = np.zeros(4, dtype=np.float32)
    for i in range(4):
        prev_edge = ordered[(i - 1) % 4] - ordered[i]
        next_edge = ordered[(i + 1) % 4] - ordered[i]
        cos_angle = np.dot(prev_edge, next_edge) / (
            np.linalg.norm(prev_edge) * np.linalg.norm(next_edge) + 1e-6
        )
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        errors[i] = abs(angle - 90.0)
    return errors


def _repair_dented_corner(canvas_quad, max_angle_error):
    """Reconstruct a single occluded corner from the other three.

    If the worst corner's angle error exceeds the tolerance, that corner is
    assumed to be dented by an overhanging object and is rebuilt under the
    parallelogram assumption: the dented corner equals the sum of its two
    neighbors minus the opposite corner. With only one bad corner the other three
    are trustworthy, so this restores the true canvas quad. If no corner is bad
    enough, the quad is returned unchanged.

    Args:
        canvas_quad: (4, 2) float32 canvas corners (any order).
        max_angle_error: Degrees of right-angle deviation above which a corner is
            treated as dented and reconstructed.

    Returns:
        A (4, 2) float32 array of corners in [TL, TR, BR, BL] order.
    """
    ordered = order_corners(canvas_quad)
    errors = _corner_angle_errors(ordered)
    worst = int(np.argmax(errors))
    if errors[worst] <= max_angle_error:
        return ordered

    # Parallelogram closure: a corner equals (its two neighbors) minus (the
    # opposite corner). Neighbors of index i are (i-1) and (i+1); opposite is
    # (i+2), all modulo 4 around the ordered quad.
    prev_corner = ordered[(worst - 1) % 4]
    next_corner = ordered[(worst + 1) % 4]
    opposite = ordered[(worst + 2) % 4]
    repaired = ordered.copy()
    repaired[worst] = prev_corner + next_corner - opposite
    return repaired


def find_sheet_corners(
    img_bgr,
    min_area_ratio=0.05,
    repair_dents=True,
    max_angle_error=8.0,
    prev_corners=None,
):
    """Detect the 4 corners of the white canvas.

    The white canvas is the measured surface and maps directly to the unit square.

    Strategy:
        1. Primary - canvas blob: Otsu isolates the bright canvas on the dark mat;
           its contour is reduced to a 4-point perspective quad. This alone is
           accurate whenever the canvas edges are unobstructed.
        2. Dent repair: if one corner's angle deviates too far from the expected
           near-right angle (an object overhanging and denting that corner), it is
           reconstructed from the other three under the parallelogram assumption.
        3. Fallback - previous frame: if detection fails outright and the caller
           supplies prev_corners, those are returned so a live loop can hold the
           last good warp through a momentary failure.

    Args:
        img_bgr: Input image in BGR color space.
        min_area_ratio: Minimum canvas area as a fraction of the whole image.
        repair_dents: If True, reconstruct a single corner dented by an overhang.
        max_angle_error: Degrees of right-angle deviation above which the worst
            corner is treated as dented and reconstructed.
        prev_corners: Optional (4, 2) corners from the previous frame, returned
            as a last resort if the current frame yields no canvas.

    Returns:
        A (4, 2) float32 array of corner points, or None (and not prev_corners)
        if no canvas could be found.
    """
    img_h, img_w = img_bgr.shape[:2]
    image_area = img_h * img_w

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = _white_mask(blurred)

    canvas_quad = _find_canvas_quad(mask, image_area, min_area_ratio)
    if canvas_quad is None:
        return prev_corners

    if repair_dents:
        canvas_quad = _repair_dented_corner(canvas_quad, max_angle_error)

    return canvas_quad


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
    """Perspective-correct the canvas to a fixed-size top-down square.

    Args:
        img_bgr: Input image in BGR color space.
        corners: A (4, 2) array of the canvas corners (any order).
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
    """Map a warped pixel coordinate to normalized (0.0-1.0) canvas coordinates.

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
