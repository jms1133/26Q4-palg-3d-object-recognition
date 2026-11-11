"""Color module: per-color HSV masking and color classification.

This stage runs on the WARPED sheet produced by the calibration module, so the
region of interest is already flat and isolated. Working in HSV lets us separate
objects by hue while rejecting two kinds of unwanted pixels:

    - the matte WHITE sheet, which has very low saturation, and
    - dark CAST SHADOWS from the 3D objects, which have very low value (V).

The thresholds below were measured from white-balanced flash captures of each
object pair (one pair per color) on the real sheet. Those measurements drove
three design choices that a naive per-color ``inRange`` cannot handle:

    1. The pastel objects are genuinely low-saturation (purple S~53, pink S~77),
       so a single high SAT_FLOOR would erase them. The floor is lowered and the
       white sheet is instead rejected by combining low saturation WITH high
       value (the sheet is both desaturated and bright; pastels are not).
    2. The dark green objects reflect very little light (V~15) but keep a correct
       high-saturation green hue. Green therefore gets its own low value floor so
       it is not discarded as shadow.
    3. Red (S~180) and pink (S~77) share almost the same hue near the 0/180 seam.
       Hue alone cannot separate them; saturation does (split at RED_PINK_SAT).

Because of these interactions, classification is expressed as an ordered set of
rules (see ``classify_color_hsv``) rather than independent hue bands. The
``COLOR_RANGES`` table is retained for building segmentation masks via
``cv2.inRange`` and is derived from the same measured bounds.

Public API:
    build_color_mask(hsv, color_name) -> cleaned binary mask for one color
    build_all_masks(hsv)              -> dict of {color_name: mask}
    classify_color_hsv(hsv_sample)    -> human-readable color label
"""

import cv2
import numpy as np

# General saturation floor for "this pixel belongs to a colored object". Set low
# so the pale pastels (purple ~53, pink ~77) survive. The white sheet is not
# rejected by this floor alone but by SHEET_SAT_MAX combined with SHEET_VALUE_MIN.
SAT_FLOOR = 30

# Default value floor for chromatic objects. Most colors sit well above this;
# dark green is the exception and uses GREEN_VALUE_FLOOR instead.
VALUE_FLOOR = 40

# The white sheet is simultaneously desaturated AND bright. A pixel is treated as
# background only when BOTH hold, so a bright-but-pale pastel is not mistaken for
# the sheet.
SHEET_SAT_MAX = 30
SHEET_VALUE_MIN = 140

# Dark green reflects little light (measured V~15) yet keeps a true, saturated
# green hue, so it needs a much lower value floor than the other colors.
GREEN_VALUE_FLOOR = 10

# Red and pink overlap in hue near the 0/180 seam and are separated by
# saturation: vivid red (~180) versus pale pink (~77).
RED_PINK_SAT = 120

# Gold needs a higher saturation floor than the other colors. The warm light cast
# on the white sheet lands in the gold hue band at low saturation (measured ~30-60)
# and would paint a false yellow halo across the background; the real yellow
# objects are vivid (S~125), so this floor keeps them while rejecting the bleed.
GOLD_SAT_FLOOR = 75

# Per-color HSV bounds as lists of (lower, upper) pairs, used to build
# segmentation masks. These mirror the measured hue bands; red and pink share
# the seam hues and are disambiguated by their saturation floors. Each color's
# own value floor is applied (green is the low-value exception).
COLOR_RANGES = {
    "red": [
        ((0, RED_PINK_SAT, VALUE_FLOOR), (12, 255, 255)),
        ((168, RED_PINK_SAT, VALUE_FLOOR), (180, 255, 255)),
    ],
    "gold": [
        ((13, GOLD_SAT_FLOOR, VALUE_FLOOR), (34, 255, 255)),
    ],
    "green": [
        ((40, SAT_FLOOR, GREEN_VALUE_FLOOR), (80, 255, 255)),
    ],
    "blue": [
        ((85, SAT_FLOOR, VALUE_FLOOR), (110, 255, 255)),
    ],
    "pink": [
        ((0, SAT_FLOOR, VALUE_FLOOR), (12, RED_PINK_SAT, 255)),
        ((160, SAT_FLOOR, VALUE_FLOOR), (180, RED_PINK_SAT, 255)),
    ],
    "purple": [
        ((120, SAT_FLOOR, VALUE_FLOOR), (160, 255, 255)),
    ],
}

# Kernel used to clean each binary mask: opening removes speckle noise, closing
# fills small interior gaps (e.g. specular highlights on glossy spheres).
_CLEAN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def white_balance(bgr, white_patch):
    """Neutralize the illumination color cast using a known-white reference.

    Warm (home) and cold (university) lighting shift measured hue and saturation,
    which would otherwise move objects out of their tuned HSV ranges. Because the
    sheet itself is matte white, a patch of empty sheet is an ideal neutral
    reference. We scale each BGR channel so the reference patch becomes neutral
    gray (equal channels); the same scaling is applied to the whole image.

    This is a simple white-patch (von Kries) correction. Note it is most reliable
    for saturated colors: near-neutral, low-saturation surfaces can have their hue
    shifted noticeably because small channel scalings swing their dominant hue.

    Args:
        bgr: Image in BGR color space to correct.
        white_patch: A reference region expected to be white. Either a small BGR
            image/crop or a length-3 per-channel mean. Its per-channel mean is
            used as the neutral target.

    Returns:
        The white-balanced image as a uint8 BGR array of the same shape.
    """
    patch = np.asarray(white_patch, dtype=np.float32).reshape(-1, 3)
    channel_means = patch.mean(axis=0)
    gray_target = channel_means.mean()

    # Per-channel gain that maps the reference patch to neutral gray.
    gains = gray_target / (channel_means + 1e-6)

    corrected = bgr.astype(np.float32) * gains
    return np.clip(corrected, 0, 255).astype(np.uint8)


def sample_white_patch(bgr, fraction=0.06):
    """Estimate a neutral-white reference from the brightest low-color pixels.

    When the caller does not have a fixed empty-sheet location, this picks a
    reference automatically: it favors bright, low-saturation pixels (the white
    sheet) and returns their mean BGR. Intended as a convenience for the
    notebook tuning cell; production code may instead sample a fixed sheet strip.

    Args:
        bgr: Image in BGR color space (ideally the warped sheet).
        fraction: Share of the brightest qualifying pixels to average.

    Returns:
        A length-3 float32 BGR mean to pass to white_balance().
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    whiteish = (s < 40) & (v > 160)

    pixels = bgr[whiteish].astype(np.float32)
    if pixels.size == 0:
        # Fall back to the global mean if no obvious white was found.
        return bgr.reshape(-1, 3).astype(np.float32).mean(axis=0)

    brightness = pixels.mean(axis=1)
    cutoff = np.quantile(brightness, 1.0 - fraction)
    brightest = pixels[brightness >= cutoff]
    return brightest.mean(axis=0)


def build_color_mask(hsv, color_name):
    """Build a cleaned binary mask selecting one color from an HSV image.

    For colors defined by several hue ranges (red), the sub-masks are combined
    with a bitwise OR. The result is then opened and closed to suppress noise
    and fill small holes before contour extraction downstream.

    Args:
        hsv: Image in HSV color space (as from cv2.cvtColor(..., COLOR_BGR2HSV)).
        color_name: Key into COLOR_RANGES (e.g. "red", "blue").

    Returns:
        A single-channel uint8 mask (0 or 255) the same size as the input.

    Raises:
        KeyError: If color_name is not a known color.
    """
    if color_name not in COLOR_RANGES:
        raise KeyError(f"Unknown color '{color_name}'. Known: {list(COLOR_RANGES)}")

    height, width = hsv.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for lower, upper in COLOR_RANGES[color_name]:
        part = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.bitwise_or(mask, part)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _CLEAN_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _CLEAN_KERNEL)
    return mask


def build_all_masks(hsv):
    """Build one cleaned mask per known color.

    Running color detection independently per color is what lets touching
    objects of DIFFERENT colors separate for free: in each color's mask, the
    other-colored object simply disappears.

    Args:
        hsv: Image in HSV color space.

    Returns:
        A dict mapping each color name in COLOR_RANGES to its binary mask.
    """
    return {name: build_color_mask(hsv, name) for name in COLOR_RANGES}


def classify_color_hsv(hsv_sample):
    """Map an HSV sample to a human-readable color label.

    Intended to be called with the MEAN (or median) HSV of an object's eroded
    interior rather than a single centroid pixel, which is noisy on textured or
    glossy surfaces.

    The rules are applied in a deliberate ORDER because the measured colors
    overlap in ways independent hue bands cannot resolve:

        1. Green is tested first on hue alone. The dark green objects have very
           low value (V~15); testing them before the shadow guard prevents them
           from being discarded as shadow.
        2. The shadow guard rejects truly black pixels.
        3. The sheet guard rejects pixels that are BOTH pale and bright (the
           white sheet). Pale-but-dim pastels survive this.
        4. Red versus pink: both sit near the 0/180 hue seam, so they are split
           by saturation (vivid red vs pale pink), not hue.
        5. The remaining hue bands (gold, blue, purple) are unambiguous.

    Args:
        hsv_sample: An (H, S, V) triple with H in [0, 180], S and V in [0, 255].

    Returns:
        One of the COLOR_RANGES keys, "white", "shadow", or "unknown".
    """
    h, s, v = (float(hsv_sample[0]), float(hsv_sample[1]), float(hsv_sample[2]))

    # 1. Green first: it is dark (low V) but keeps a saturated green hue, so it
    #    must be caught before the shadow guard would reject it.
    if 40 <= h <= 80 and s >= SAT_FLOOR and v >= GREEN_VALUE_FLOOR:
        return "green"

    # 2. Truly black pixels are shadow.
    if v < VALUE_FLOOR:
        return "shadow"

    # 3. The white sheet is pale AND bright; pastels are pale but not bright.
    if s < SHEET_SAT_MAX and v >= SHEET_VALUE_MIN:
        return "white"
    if s < SAT_FLOOR:
        return "white"

    # 4. Red/pink share the hue seam near 0/180; separate them by saturation.
    if h <= 12 or h >= 160:
        return "red" if s >= RED_PINK_SAT else "pink"

    # 5. Unambiguous hue bands. Gold uses a higher saturation floor to reject the
    #    warm cast that the white sheet shows in the gold hue band.
    if 13 <= h <= 34:
        return "gold" if s >= GOLD_SAT_FLOOR else "white"
    if 85 <= h <= 110:
        return "blue"
    if 120 <= h <= 160:
        return "purple"
    return "unknown"
