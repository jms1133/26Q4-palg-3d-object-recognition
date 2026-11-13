"""Segmentation module: separate SAME-color objects that touch side by side.

Per-color masking (see ``color.build_all_masks``) already separates touching
objects of DIFFERENT colors for free: in each color's mask the other-colored
object simply disappears. What it cannot handle is two or more objects of the
SAME color resting against each other, which merge into one contour. This module
splits such merged blobs so each physical object is reported separately.

Approach (distance transform + watershed):
    1. The distance transform of a blob peaks at the center of each object and
       dips along the thin bridge where two objects touch.
    2. Local maxima of that transform give one seed per object. Seeds closer than
       a minimum separation are merged (keeping the stronger one) to suppress the
       several near-equal peaks that a single chunky shape can produce.
    3. The seeds label markers for ``cv2.watershed``, which floods the blob and
       draws boundaries along the touch bridges, yielding one region per object.

Isolated objects produce a single seed, so the split is a no-op for them and the
original contour is returned unchanged.

Limitation: watershed can still over-segment large, chunky 3D prisms whose top
faces create several distance-transform peaks of similar height. The minimum-
separation merge reduces this but does not eliminate it for every arrangement;
objects of clearly different sizes touching are the hardest case. Touching
objects are nonetheless handled for the common cases, as required.

Public API:
    separate_contour(blob_mask) -> list of per-object contours
    segment_color_mask(mask)    -> flattened list of per-object contours
"""

import cv2
import numpy as np

from . import shapes

# Distance-transform peak floor as a fraction of the blob's maximum distance.
# Pixels below this are too near the border to be an object center.
MIN_DISTANCE_FRACTION = 0.4

# Side length of the dilation window used to find local maxima of the distance
# transform. One maximum per window suppresses noisy near-duplicate peaks while
# still resolving separate object centers.
PEAK_WINDOW = 31

# Seeds closer than this (in warped pixels) are merged into one, keeping the seed
# on the higher distance value. This removes the several near-equal peaks a single
# chunky object can generate without merging genuinely distinct object centers.
MIN_SEED_SEPARATION = 50

# Small kernel to coalesce adjacent peak pixels into single connected seeds.
_SEED_MERGE_KERNEL = np.ones((3, 3), dtype=np.uint8)


def _find_seeds(blob_mask):
    """Find one seed point per object inside a single-blob mask.

    Args:
        blob_mask: Binary (0/255) mask containing exactly one filled blob.

    Returns:
        A list of (x, y) integer seed points, ordered strongest first.
    """
    dist = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
    max_dist = dist.max()
    if max_dist <= 0:
        return []

    # Local maxima: pixels equal to the window maximum and above the floor.
    window = np.ones((PEAK_WINDOW, PEAK_WINDOW), dtype=np.uint8)
    local_max = cv2.dilate(dist, window)
    is_peak = (dist >= local_max - 1e-3) & (dist > MIN_DISTANCE_FRACTION * max_dist)
    peak_mask = is_peak.astype(np.uint8) * 255

    # Coalesce adjacent peak pixels, then take each component's centroid.
    peak_mask = cv2.dilate(peak_mask, _SEED_MERGE_KERNEL)
    count, _, _, centroids = cv2.connectedComponentsWithStats(peak_mask)
    points = [(int(centroids[i][0]), int(centroids[i][1])) for i in range(1, count)]

    # Merge seeds closer than MIN_SEED_SEPARATION, keeping the higher-distance one.
    points.sort(key=lambda p: -dist[p[1], p[0]])
    kept = []
    for p in points:
        far_enough = all(
            (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= MIN_SEED_SEPARATION ** 2
            for q in kept
        )
        if far_enough:
            kept.append(p)
    return kept


def _contours_from_labels(markers, label_count):
    """Extract one external contour per watershed label.

    Args:
        markers: Int32 marker image returned by cv2.watershed.
        label_count: Number of object labels (markers use 2..label_count+1).

    Returns:
        A list of contours, one per label region above MIN_CONTOUR_AREA.
    """
    result = []
    # Watershed object labels start at 2 (1 is the background seed, -1 borders).
    for label in range(2, label_count + 2):
        region = (markers == label).astype(np.uint8) * 255
        if cv2.countNonZero(region) == 0:
            continue
        contours, _ = cv2.findContours(
            region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            if cv2.contourArea(contour) >= shapes.MIN_CONTOUR_AREA:
                result.append(contour)
    return result


def separate_contour(blob_mask):
    """Split one same-color blob into its constituent object contours.

    Args:
        blob_mask: Binary (0/255) mask containing exactly one filled blob (for
            example a single contour drawn filled onto a blank image).

    Returns:
        A list of object contours. If only one object is detected the blob's own
        single external contour is returned, so isolated objects pass through
        unchanged.
    """
    seeds = _find_seeds(blob_mask)

    # Zero or one seed: nothing to split, return the blob's own contour(s).
    if len(seeds) <= 1:
        contours, _ = cv2.findContours(
            blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return [c for c in contours if cv2.contourArea(c) >= shapes.MIN_CONTOUR_AREA]

    # Label a marker image: background as 1, each seed as its own id from 2 up.
    markers = np.zeros(blob_mask.shape, dtype=np.int32)
    markers[blob_mask == 0] = 1
    for index, (x, y) in enumerate(seeds, start=2):
        markers[y, x] = index

    # Watershed needs a 3-channel image; the blob itself is enough structure.
    color_blob = cv2.cvtColor(blob_mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color_blob, markers)

    return _contours_from_labels(markers, len(seeds))


def segment_color_mask(mask):
    """Split every merged blob in a color mask into per-object contours.

    This is the step the main loop runs on each color mask before shape and
    position classification, so that same-color touching objects are reported
    individually.

    Args:
        mask: A cleaned binary (0/255) mask for one color (from
            color.build_color_mask).

    Returns:
        A flattened list of per-object contours across all blobs in the mask,
        each at or above shapes.MIN_CONTOUR_AREA.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    objects = []
    for contour in contours:
        if cv2.contourArea(contour) < shapes.MIN_CONTOUR_AREA:
            continue
        blob = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(blob, [contour], -1, 255, thickness=cv2.FILLED)
        objects.extend(separate_contour(blob))
    return objects
