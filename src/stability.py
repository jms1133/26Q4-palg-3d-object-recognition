"""Stability module: decide when the scene is static enough to analyze.

The live loop should classify objects only when the scene has settled, not while
a hand is reaching in to place or move an object. Analyzing mid-motion wastes
work and makes on-screen labels flicker and jump. The project specification also
requires re-analysis to be triggered by scene changes (objects added, moved, or
removed) rather than running every frame.

Approach (frame differencing):
    1. Each incoming frame is reduced to a small, blurred grayscale image. The
       downscale and blur suppress sensor noise and sub-pixel jitter so only real
       motion registers.
    2. The absolute difference against the previous reduced frame is thresholded;
       the fraction of changed pixels is the motion score.
    3. A simple state machine latches the scene as STABLE once the motion score
       has stayed below a floor for a number of consecutive frames, and flips back
       to MOVING as soon as motion rises above the floor again. The transition
       into STABLE is reported exactly once so the caller can run classification a
       single time per settling, then hold the result until the next change.

Public API:
    StabilityDetector.update(frame_bgr) -> StabilityState
"""

from dataclasses import dataclass

import cv2
import numpy as np

# Side length the frame is downscaled to before differencing. Small enough to
# ignore noise and sub-pixel jitter, large enough to catch a hand or an object
# being placed.
PROCESS_SIZE = 120

# Per-pixel intensity change (0-255) above which a pixel counts as "changed".
# Below this, differences are treated as camera noise rather than motion.
PIXEL_DELTA_THRESHOLD = 25

# Fraction of changed pixels at or below which the frame is considered still.
# Tuned so that small lighting ripples do not count as motion but a hand does.
MOTION_FRACTION_FLOOR = 0.01

# Consecutive still frames required before the scene is declared stable. This
# debounce prevents a brief pause mid-motion from triggering a premature analysis.
STABLE_FRAMES_REQUIRED = 8


@dataclass
class StabilityState:
    """Result of a single stability update.

    Attributes:
        is_stable: True while the scene is currently settled.
        just_settled: True on the single frame the scene transitions into
            stable. The caller should run (re-)classification on this frame.
        motion: The latest motion score (fraction of changed pixels), for
            optional on-screen display or tuning.
    """

    is_stable: bool
    just_settled: bool
    motion: float


class StabilityDetector:
    """Frame-differencing state machine that reports when the scene is static.

    Feed every captured frame to ``update``. It returns a StabilityState whose
    ``just_settled`` flag fires exactly once each time the scene comes to rest,
    which is the moment to run object classification.
    """

    def __init__(
        self,
        process_size=PROCESS_SIZE,
        pixel_delta_threshold=PIXEL_DELTA_THRESHOLD,
        motion_fraction_floor=MOTION_FRACTION_FLOOR,
        stable_frames_required=STABLE_FRAMES_REQUIRED,
    ):
        """Configure the detector.

        Args:
            process_size: Side length the frame is downscaled to before diffing.
            pixel_delta_threshold: Intensity change above which a pixel counts as
                changed.
            motion_fraction_floor: Changed-pixel fraction at or below which a
                frame is considered still.
            stable_frames_required: Consecutive still frames needed to latch the
                scene as stable.
        """
        self._process_size = process_size
        self._pixel_delta_threshold = pixel_delta_threshold
        self._motion_fraction_floor = motion_fraction_floor
        self._stable_frames_required = stable_frames_required

        self._prev = None
        self._still_streak = 0
        self._is_stable = False

    def _reduce(self, frame_bgr):
        """Downscale and blur a frame to a small grayscale image for diffing.

        Args:
            frame_bgr: Input frame in BGR color space.

        Returns:
            A small single-channel uint8 image.
        """
        small = cv2.resize(
            frame_bgr, (self._process_size, self._process_size),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def _motion_score(self, reduced):
        """Fraction of pixels that changed since the previous reduced frame.

        Args:
            reduced: The current reduced frame from ``_reduce``.

        Returns:
            A float in [0.0, 1.0]; 0.0 on the very first frame.
        """
        if self._prev is None:
            return 0.0
        diff = cv2.absdiff(reduced, self._prev)
        _, changed = cv2.threshold(
            diff, self._pixel_delta_threshold, 255, cv2.THRESH_BINARY
        )
        return float(cv2.countNonZero(changed)) / changed.size

    def update(self, frame_bgr):
        """Process one frame and report the current stability state.

        Args:
            frame_bgr: The latest captured frame in BGR color space.

        Returns:
            A StabilityState describing the scene after this frame.
        """
        reduced = self._reduce(frame_bgr)
        motion = self._motion_score(reduced)
        self._prev = reduced

        just_settled = False
        if motion <= self._motion_fraction_floor:
            self._still_streak += 1
            # Latch stable only after enough consecutive still frames, and fire
            # just_settled exactly on the latching frame.
            if (
                not self._is_stable
                and self._still_streak >= self._stable_frames_required
            ):
                self._is_stable = True
                just_settled = True
        else:
            # Any real motion breaks the streak and marks the scene as moving.
            self._still_streak = 0
            self._is_stable = False

        return StabilityState(
            is_stable=self._is_stable,
            just_settled=just_settled,
            motion=motion,
        )

    def reset(self):
        """Forget history so the next frame starts a fresh settling cycle."""
        self._prev = None
        self._still_streak = 0
        self._is_stable = False
