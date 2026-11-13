"""Shape module: classify object shape and report shape + color + position.

This stage runs on contours extracted from the per-color masks produced by the
``color`` module (see ``color.build_all_masks``). Because masking is done per
color, each contour here already belongs to a single color; this module adds the
object's SHAPE and its normalized POSITION to complete the required
(shape, color, position) description for each object.

The object set is polygonal: triangles/pyramids, quadrilaterals (cubes and
cuboids imaged as four-sided tops/faces) and pentagons (pentagon-prisms). A
``circle`` branch is kept for a possible sphere or cylinder top, which images as
a near-circular contour, even though the current set contains none.

Classification combines several contour features rather than relying on vertex
count alone, because the polygon approximation is unstable on its own. The
measured features that separate the shapes are:

    - vertices: ``approxPolyDP`` corner count (3 triangle, 4 quad, 5 pentagon).
    - circularity ``4*pi*Area / Perimeter^2``: high for round blobs, moderate for
      filled polygons, lower for thin triangles.
    - solidity ``Area / ConvexHullArea``: reported for diagnostics. It is NOT used
      as a classification guard: a legitimate 3D pentagon-prism shows concave side
      faces (measured solidity ~0.70) while a blob of merged same-color cubes can
      stay fairly convex (~0.88), so solidity cannot distinguish the two. Splitting
      merged same-color objects is deferred to the touching-object separation step,
      which runs before this classifier in the final pipeline.

Public API:
    shape_metrics(contour)              -> dict of raw geometric features
    classify_shape(contour)             -> shape label string
    describe_object(contour, hsv, size) -> dict with shape, color, position, area
"""

import cv2
import numpy as np

from . import calibration
from . import color

# Contours smaller than this (in warped pixels) are treated as noise and not
# classified. The smallest real objects measured ~5000 px; speckle is far below.
MIN_CONTOUR_AREA = 1500

# Circularity at or above this is considered round (a sphere or cylinder top).
# Filled polygons in the set measured <= ~0.71, so this cleanly excludes them.
CIRCLE_CIRCULARITY = 0.80

# approxPolyDP epsilon as a fraction of the contour perimeter. Measured on real
# objects, this tolerance gives stable corner counts (3/4/5) across the set. At
# 0.05 the 3D pentagon-prisms reduce cleanly to 5 corners (a smaller tolerance
# leaves a spurious 6th vertex on a visible top edge) while triangles stay at 3
# and quadrilaterals at 4.
APPROX_EPS_FACTOR = 0.05

# Kernel used to erode an object's mask to an interior core before sampling its
# color, so glossy tops and shadowed edges contribute less (per CLAUDE.md 5D).
_INTERIOR_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Map a stable vertex count to its polygon label.
_VERTEX_SHAPES = {3: "triangle", 4: "quadrilateral", 5: "pentagon"}


def shape_metrics(contour):
    """Compute the geometric features used to classify a contour.

    Args:
        contour: An OpenCV contour (Nx1x2 int array).

    Returns:
        A dict with:
            area: Contour area in pixels.
            perimeter: Contour arc length (closed).
            circularity: 4*pi*Area / Perimeter^2 (0 if perimeter is 0).
            aspect_ratio: Bounding-box width / height (0 if height is 0).
            solidity: Area / convex-hull area (0 if hull area is 0).
            vertices: approxPolyDP corner count at APPROX_EPS_FACTOR.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    circularity = 4.0 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0

    _, _, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h) if h > 0 else 0.0

    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area > 0 else 0.0

    approx = cv2.approxPolyDP(contour, APPROX_EPS_FACTOR * perimeter, True)
    vertices = len(approx)

    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "vertices": vertices,
    }


def classify_shape(contour):
    """Classify a single contour into a shape label.

    The rules are applied in order:
        1. Reject contours below MIN_CONTOUR_AREA as noise ("unknown").
        2. A high-circularity contour is a "circle" (sphere / cylinder top).
        3. Otherwise map the approxPolyDP vertex count to a polygon label
           (3 triangle, 4 quadrilateral, 5 pentagon). Any other count is
           "unknown".

    Separating same-color objects that touch and merge into one contour is NOT
    done here (solidity cannot reliably distinguish such a blob from a legitimate
    3D prism); it is handled by the touching-object separation step that runs
    before this classifier in the final pipeline.

    Args:
        contour: An OpenCV contour (Nx1x2 int array).

    Returns:
        One of "triangle", "quadrilateral", "pentagon", "circle", or "unknown".
    """
    m = shape_metrics(contour)

    if m["area"] < MIN_CONTOUR_AREA:
        return "unknown"

    if m["circularity"] >= CIRCLE_CIRCULARITY:
        return "circle"

    return _VERTEX_SHAPES.get(m["vertices"], "unknown")


def sample_contour_color(contour, hsv):
    """Return the color label of a contour's eroded interior.

    The contour's own filled mask is eroded to an interior core, and the median
    HSV over that core is passed to the production color classifier. Eroding
    avoids edge and highlight pixels, giving a stable body color.

    Args:
        contour: An OpenCV contour (Nx1x2 int array).
        hsv: The warped image in HSV color space the contour was found in.

    Returns:
        A color label from color.classify_color_hsv, or "unknown" if the eroded
        interior is empty.
    """
    height, width = hsv.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    mask = cv2.erode(mask, _INTERIOR_KERNEL)

    pixels = hsv[mask > 0]
    if pixels.shape[0] == 0:
        return "unknown"

    median = np.median(pixels, axis=0)
    return color.classify_color_hsv(median)


def object_centroid(contour):
    """Return the pixel centroid (cx, cy) of a contour via image moments.

    Args:
        contour: An OpenCV contour (Nx1x2 int array).

    Returns:
        An (cx, cy) integer pixel tuple, or None if the contour has zero area.
    """
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return cx, cy


def describe_object(contour, hsv, size):
    """Build the full (shape, color, position) description for one contour.

    This is the per-object record the main loop overlays and prints. The position
    is the centroid mapped to normalized canvas coordinates, with (0, 0) at the
    top-left and (1, 1) at the bottom-right per the project specification.

    Args:
        contour: An OpenCV contour (Nx1x2 int array) from a per-color mask.
        hsv: The warped image in HSV color space the contour was found in.
        size: Side length in pixels of the warped square (for normalization).

    Returns:
        A dict with keys 'shape', 'color', 'position' (a (norm_x, norm_y) tuple
        or None), 'centroid' (pixel (cx, cy) or None) and 'area', or None if the
        contour has zero area.
    """
    centroid = object_centroid(contour)
    if centroid is None:
        return None

    norm = calibration.pixel_to_normalized(centroid[0], centroid[1], size)

    return {
        "shape": classify_shape(contour),
        "color": sample_contour_color(contour, hsv),
        "position": norm,
        "centroid": centroid,
        "area": cv2.contourArea(contour),
    }
