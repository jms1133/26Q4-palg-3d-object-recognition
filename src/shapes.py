"""Shape module: classify object shape and report shape + color + position.

This stage runs on contours extracted from the per-color masks produced by the
``color`` module (see ``color.build_all_masks``). Because masking is done per
color, each contour here already belongs to a single color; this module adds the
object's SHAPE and its normalized POSITION to complete the required
(shape, color, position) description for each object.

The object set is polygonal: triangles/pyramids, four-sided tops split into
squares (cubes) and rectangles (cuboids), and pentagons (pentagon-prisms). A
``circle`` branch is kept for a possible sphere or cylinder top, which images as
a near-circular contour, even though the current set contains none.

Classification combines several contour features rather than relying on vertex
count alone, because the polygon approximation is unstable on its own. The
measured features that separate the shapes are:

    - vertices: ``approxPolyDP`` corner count (3 triangle, 4 four-sided,
      5 pentagon).
    - rotated aspect ratio ``minAreaRect`` long/short side: splits a four-sided
      top into square (cube, ~1.0) and rectangle (cuboid, ~1.5). It is rotation-
      invariant, unlike the axis-aligned bounding box, so it holds for tilted 3D
      objects.
    - circularity ``4*pi*Area / Perimeter^2``: high for round blobs, moderate for
      filled polygons, lower for thin triangles.
    - solidity ``Area / ConvexHullArea``: reported for diagnostics. It is NOT used
      as a classification guard: a legitimate 3D pentagon-prism shows concave side
      faces (measured solidity ~0.70) while a blob of merged same-color cubes can
      stay fairly convex (~0.88), so solidity cannot distinguish the two. Splitting
      merged same-color objects is deferred to the touching-object separation step,
      which runs before this classifier in the final pipeline.

Public API:
    shape_metrics(contour)                     -> dict of raw geometric features
    classify_shape(contour)                    -> shape label string
    constrain_shape_to_color(shape, vertices, color) -> shape corrected by color prior
    describe_object(contour, hsv, size)        -> dict with shape, color, position, area
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

# Rotation-invariant aspect ratio (minAreaRect long side / short side) at or above
# which a four-sided top face is a rectangle (cuboid) rather than a square (cube).
# Measured on warped captures: cube tops sit near 1.0-1.1 and cuboid tops near
# 1.4-1.5, so a 1.25 split cleanly separates them. The axis-aligned bounding box
# cannot be used here because 3D objects image rotated; the rotated rectangle is
# orientation-independent.
SQUARE_MAX_ASPECT = 1.25

# Kernel used to erode an object's mask to an interior core before sampling its
# color, so glossy tops and shadowed edges contribute less (per CLAUDE.md 5D).
_INTERIOR_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Canonical vertex count for each polygon label. Square and rectangle are BOTH
# four-sided tops: vertex count cannot tell them apart, so they share count 4 and
# are separated later by aspect ratio (see classify_shape / SQUARE_MAX_ASPECT).
_SHAPE_VERTICES = {
    "triangle": 3,
    "square": 4,
    "rectangle": 4,
    "pentagon": 5,
}

# Shapes that physically exist in each color of the object set. This domain prior
# corrects the dominant failure mode of imaging 3D objects: a block's darker side
# faces and cast shadows are captured alongside its top face, so the silhouette
# gains spurious corners (a cuboid reads as 5-6 vertices, a shaded triangle as 4).
# Because these extra corners are physics, not noise we can threshold away, each
# raw vertex count is snapped to the NEAREST vertex count the color actually
# admits. Colors are only ever mapped onto their own real shapes, so a genuine
# triangle or pentagon is never invented.
#
# Square (cube) vs rectangle (cuboid) share vertex count 4. In the object set no
# single color owns BOTH: blue and yellow have only squares, red and green have only
# rectangles. So the color prior alone resolves which four-sided shape an object
# is, and the aspect ratio only matters as a tie-break for a color that admits
# both (none currently). This keeps square/rectangle robust without leaning on a
# single aspect-ratio threshold.
ALLOWED_SHAPES = {
    "blue": {"square"},
    "red": {"triangle", "rectangle"},
    "yellow": {"square", "pentagon"},
    "green": {"triangle", "pentagon", "rectangle"},
}


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
            rotated_aspect: minAreaRect long side / short side (>= 1.0, or 0 if
                the short side is 0). Orientation-independent, so it separates a
                square (cube) top from a rectangle (cuboid) top even when the
                object images rotated.
            solidity: Area / convex-hull area (0 if hull area is 0).
            vertices: approxPolyDP corner count at APPROX_EPS_FACTOR.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    circularity = 4.0 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0

    _, _, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h) if h > 0 else 0.0

    (_, (rw, rh), _) = cv2.minAreaRect(contour)
    long_side, short_side = max(rw, rh), min(rw, rh)
    rotated_aspect = long_side / short_side if short_side > 0 else 0.0

    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area > 0 else 0.0

    approx = cv2.approxPolyDP(contour, APPROX_EPS_FACTOR * perimeter, True)
    vertices = len(approx)

    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
        "rotated_aspect": rotated_aspect,
        "solidity": solidity,
        "vertices": vertices,
    }


def _quad_label(rotated_aspect):
    """Label a four-sided top as "square" or "rectangle" by its aspect ratio.

    A cube top images as a near-square (rotated aspect ~1.0-1.1) and a cuboid top
    as an elongated rectangle (~1.4-1.5), so SQUARE_MAX_ASPECT splits them.

    Args:
        rotated_aspect: The minAreaRect long/short side ratio (>= 1.0).

    Returns:
        "square" if the ratio is below SQUARE_MAX_ASPECT, else "rectangle".
    """
    return "square" if rotated_aspect < SQUARE_MAX_ASPECT else "rectangle"


def classify_shape(contour):
    """Classify a single contour into a shape label.

    The rules are applied in order:
        1. Reject contours below MIN_CONTOUR_AREA as noise ("unknown").
        2. A high-circularity contour is a "circle" (sphere / cylinder top).
        3. Map the approxPolyDP vertex count to a polygon family
           (3 triangle, 4 four-sided, 5 pentagon). Any other count is "unknown".
        4. A four-sided top is further split into "square" (cube) or "rectangle"
           (cuboid) by its rotation-invariant aspect ratio.

    Separating same-color objects that touch and merge into one contour is NOT
    done here (solidity cannot reliably distinguish such a blob from a legitimate
    3D prism); it is handled by the touching-object separation step that runs
    before this classifier in the final pipeline.

    Args:
        contour: An OpenCV contour (Nx1x2 int array).

    Returns:
        One of "triangle", "square", "rectangle", "pentagon", "circle", or
        "unknown".
    """
    m = shape_metrics(contour)

    if m["area"] < MIN_CONTOUR_AREA:
        return "unknown"

    if m["circularity"] >= CIRCLE_CIRCULARITY:
        return "circle"

    if m["vertices"] == 3:
        return "triangle"
    if m["vertices"] == 4:
        return _quad_label(m["rotated_aspect"])
    if m["vertices"] == 5:
        return "pentagon"
    return "unknown"


def constrain_shape_to_color(shape, vertices, color_name, rotated_aspect=None):
    """Snap a raw shape to the nearest shape the color can physically be.

    This corrects the dominant 3D-imaging artifact described on ALLOWED_SHAPES:
    shaded side faces and shadows add spurious corners, so a cuboid can read as 5-6
    vertices and a shaded triangle as 4. Using the raw approxPolyDP vertex count,
    the object is mapped to whichever allowed shape has the nearest canonical vertex
    count, with ties broken toward fewer vertices so a clean triangle is never
    pulled up to a four-sided shape.

    Square (cube) and rectangle (cuboid) both have four corners, so vertex count
    alone cannot tell them apart. When the snap lands on the four-sided family and
    the color admits BOTH, the aspect ratio decides; if it admits only one, that
    one is used. A four-sided artifact (e.g. a 6-vertex 3D cuboid) is therefore
    still resolved to the correct square/rectangle.

    A "circle" is left untouched (roundness, not a vertex count, defines it). For a
    color with no known allowed set, the raw label is returned unchanged.

    Args:
        shape: The raw shape label from classify_shape.
        vertices: The raw approxPolyDP vertex count for the same contour.
        color_name: The object's classified color.
        rotated_aspect: The contour's rotation-invariant aspect ratio, used to
            pick square vs rectangle. Falls back to SQUARE_MAX_ASPECT (square) if
            not provided.

    Returns:
        The possibly corrected shape label.
    """
    if shape == "circle":
        return shape

    allowed = ALLOWED_SHAPES.get(color_name)
    if not allowed:
        return shape

    if shape in allowed:
        return shape

    # Snap to the allowed shape whose canonical vertex count is closest to the
    # measured count; break ties toward fewer vertices (the simpler shape).
    target = min(
        allowed,
        key=lambda name: (abs(_SHAPE_VERTICES[name] - vertices), _SHAPE_VERTICES[name]),
    )

    # If the snap landed on a four-sided shape and the color admits BOTH square
    # and rectangle, the aspect ratio (not vertex count) decides which one.
    if target in ("square", "rectangle") and {"square", "rectangle"} <= allowed:
        aspect = SQUARE_MAX_ASPECT if rotated_aspect is None else rotated_aspect
        return _quad_label(aspect)

    return target


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

    object_color = sample_contour_color(contour, hsv)
    metrics = shape_metrics(contour)
    raw_shape = classify_shape(contour)
    object_shape = constrain_shape_to_color(
        raw_shape, metrics["vertices"], object_color, metrics["rotated_aspect"]
    )

    return {
        "shape": object_shape,
        "color": object_color,
        "position": norm,
        "centroid": centroid,
        "area": cv2.contourArea(contour),
    }
