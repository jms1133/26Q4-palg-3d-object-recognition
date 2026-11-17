"""Exposure module: optional, multi-platform hardware exposure control.

Why this exists:
    With the small-black-paper-on-white-desk setup, the camera's auto-exposure
    averages the scene and overexposes the white canvas, washing out colors. Locking
    the exposure to the canvas highlights keeps colors saturated. The mechanism to
    do that differs per operating system:

    - macOS: OpenCV's ``cap.set(cv2.CAP_PROP_EXPOSURE, ...)`` is ignored because
      Apple's AVFoundation does not pass USB Video Class (UVC) property writes
      through to the device. We instead shell out to ``uvc-util-app``, a native
      macOS helper compiled into the project root, which writes UVC controls.
    - Linux: OpenCV's V4L2 backend DOES honor ``CAP_PROP_AUTO_EXPOSURE`` and
      ``CAP_PROP_EXPOSURE``, so we can set exposure directly through ``cap.set``.

Design constraints (project spec):
    - The application must run completely OFFLINE: this module only invokes a local
      binary (macOS) or OpenCV property writes (Linux); no network communication.
    - Every entry point is FAIL-SAFE. If the platform is unsupported, the helper is
      missing, or any write is rejected by the driver/camera, the controller reports
      ``available == False`` and all methods become harmless no-ops returning None.
      The caller then runs with the camera's default auto-exposure and the rest of
      the pipeline is unaffected. This matters because the units and the
      auto-exposure magic values are driver/camera-specific and cannot be verified
      on every grading machine; the feature is best-effort, never required.

Public API:
    make_exposure_controller(camera_index, cap) -> ExposureController
        Returns the right backend for this OS (or a disabled no-op controller).
    ExposureController interface (shared by all backends):
        .available      -> bool: whether manual control is usable on this host.
        .calibrate(cap) -> int|None: lock exposure so canvas whites sit ~target.
        .nudge(step)    -> int|None: shift exposure by +/- step, clamped.
        .value          -> int|None: last applied exposure, or None.
"""

import platform
import subprocess

import cv2
import numpy as np

# Name of the native macOS helper expected in the project root. It is invoked as a
# relative path, so the application must be started from the project root (as the
# README instructs). Kept as a module constant so there is a single source of truth.
UVC_UTIL_BINARY = "./uvc-util-app"

# macOS UVC exposure-time range reported by the Aukey webcam used in this project.
# Values are device-specific UVC units (not seconds); they are the empirically
# confirmed min/max for this camera.
MIN_EXPOSURE = 1
MAX_EXPOSURE = 5000

# Linux/V4L2 exposure range. OpenCV's CAP_PROP_EXPOSURE scale is driver-specific
# and differs from the macOS UVC units, so Linux uses its own conservative bounds
# and starting guess. These are best-effort defaults; the fail-safe path covers
# cameras that reject them or use a different scale.
V4L2_MIN_EXPOSURE = 1
V4L2_MAX_EXPOSURE = 5000
V4L2_DEFAULT_EXPOSURE = 150

# V4L2 CAP_PROP_AUTO_EXPOSURE magic values used by most UVC drivers:
# 1 (or 0.25 on some builds) = manual, 3 (or 0.75) = aperture-priority/auto. We try
# the integer convention first; if the driver ignores it the controller disables
# itself via the fail-safe path.
V4L2_AUTO_EXPOSURE_MANUAL = 1

# Starting guess and search parameters for highlight-priority calibration (shared).
DEFAULT_EXPOSURE = 150
TARGET_WHITE_LEVEL = 235      # desired brightness of the canvas whites (0-255)
WHITE_TOLERANCE = 10          # accepted band around the target
STEP_SIZE = 10                # exposure units changed per calibration iteration
BRIGHT_PERCENTILE = 95        # measure only the brightest pixels (the white canvas)
OVEREXPOSED_LEVEL = 250       # above this, back off twice as fast
MAX_CALIBRATION_STEPS = 60    # hard cap so calibration always terminates

# Subprocess timeout (seconds) so a misbehaving binary can never hang the loop.
_SUBPROCESS_TIMEOUT = 2.0


class ExposureController:
    """Base controller: shared, backend-agnostic calibration logic.

    Subclasses implement the three hardware hooks (``_probe``, ``_enter_manual_mode``,
    ``_write_exposure``) and the device-specific range via ``min_exposure`` /
    ``max_exposure`` / ``default_exposure``. This base provides the highlight-priority
    search loop, the clamp/cache, and the ``nudge`` helper, so all backends behave
    identically and remain fail-safe (``available`` gates every public method).

    The default (base) instance is a disabled no-op controller, returned by the
    factory on unsupported platforms.
    """

    min_exposure = MIN_EXPOSURE
    max_exposure = MAX_EXPOSURE
    default_exposure = DEFAULT_EXPOSURE

    def __init__(self, camera_index=0):
        """Probe the environment once and cache whether manual control is usable.

        Args:
            camera_index: The capture device index for this controller's backend.
        """
        self.camera_index = camera_index
        self.value = None
        self.available = self._probe()

    # --- hooks overridden by backends -------------------------------------------

    def _probe(self):
        """Return True if manual control is usable on this host. Base: never."""
        return False

    def _enter_manual_mode(self):
        """Switch the camera to manual exposure mode. Return True on success."""
        return False

    def _write_exposure(self, exposure):
        """Write a clamped exposure value to the device. Return True on success.

        Args:
            exposure: Integer exposure already clamped to the backend's range.
        """
        return False

    # --- shared logic -----------------------------------------------------------

    def _apply_exposure(self, exposure):
        """Clamp to the backend range, write it, and cache it on success.

        Args:
            exposure: Desired exposure value in the backend's device units.

        Returns:
            The clamped value that was applied, or None if the write failed.
        """
        exposure = int(max(self.min_exposure, min(self.max_exposure, exposure)))
        if self._write_exposure(exposure):
            self.value = exposure
            return exposure
        return None

    @staticmethod
    def _white_level(frame):
        """Brightness of the canvas highlights: the Nth percentile of gray values.

        Measuring a high percentile (rather than the mean) targets the white canvas
        and ignores the dark border/mat, which would otherwise drag the average down
        and cause overexposure.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.percentile(gray, BRIGHT_PERCENTILE))

    def calibrate(self, cap):
        """Lock exposure so the white canvas sits near ``TARGET_WHITE_LEVEL``.

        Iteratively nudges the exposure down/up based on the measured highlight
        level until it falls within tolerance, hits a range limit, or the step cap
        is reached (so this always terminates).

        Args:
            cap: The already-opened ``cv2.VideoCapture`` for the same device.

        Returns:
            The locked exposure value, or None if manual control is unavailable.
        """
        if not self.available:
            return None
        if not self._enter_manual_mode():
            self.available = False
            return None

        # Let the sensor settle after switching to manual mode.
        for _ in range(5):
            cap.read()

        exposure = self.default_exposure
        for _ in range(MAX_CALIBRATION_STEPS):
            if self._apply_exposure(exposure) is None:
                self.available = False
                return None

            # Give the hardware a few frames to reflect the new exposure.
            for _ in range(3):
                cap.read()
            ok, frame = cap.read()
            if not ok:
                break

            white = self._white_level(frame)
            print(f"Exposure {exposure} -> canvas whites {white:.1f}/255")

            if white >= OVEREXPOSED_LEVEL:
                exposure -= STEP_SIZE * 2
            elif white > TARGET_WHITE_LEVEL + WHITE_TOLERANCE:
                exposure -= STEP_SIZE
            elif white < TARGET_WHITE_LEVEL - WHITE_TOLERANCE:
                exposure += STEP_SIZE
            else:
                print(f"Exposure locked at {exposure}.")
                return exposure

            if exposure <= self.min_exposure or exposure >= self.max_exposure:
                print(f"Hit exposure limit; locked at {self.value}.")
                return self.value

        print(f"Calibration step cap reached; locked at {self.value}.")
        return self.value

    def nudge(self, step):
        """Shift the current exposure by ``step`` units (clamped to range).

        Args:
            step: Signed change in exposure units (negative = darker).

        Returns:
            The new exposure value, or None if unavailable / the write failed.
        """
        if not self.available:
            return None
        base = self.value if self.value is not None else self.default_exposure
        return self._apply_exposure(base + step)


class UvcUtilExposureController(ExposureController):
    """macOS backend: writes UVC controls via the ``uvc-util-app`` helper binary."""

    min_exposure = MIN_EXPOSURE
    max_exposure = MAX_EXPOSURE
    default_exposure = DEFAULT_EXPOSURE

    def _probe(self):
        """Return True only if the helper binary exists and responds."""
        try:
            subprocess.run(
                [UVC_UTIL_BINARY, "-I", str(self.camera_index),
                 "--show-control=exposure-time-abs"],
                capture_output=True,
                timeout=_SUBPROCESS_TIMEOUT,
                check=True,
            )
            return True
        except (OSError, subprocess.SubprocessError):
            return False

    def _set_control(self, assignment):
        """Apply one ``name=value`` UVC control; return True on success.

        Args:
            assignment: A ``control-name=value`` string for the helper's ``-s`` flag.
        """
        try:
            subprocess.run(
                [UVC_UTIL_BINARY, "-I", str(self.camera_index), "-s", assignment],
                capture_output=True,
                timeout=_SUBPROCESS_TIMEOUT,
                check=True,
            )
            return True
        except (OSError, subprocess.SubprocessError):
            return False

    def _enter_manual_mode(self):
        return self._set_control("auto-exposure-mode=1")

    def _write_exposure(self, exposure):
        return self._set_control(f"exposure-time-abs={exposure}")


class V4L2ExposureController(ExposureController):
    """Linux backend: writes exposure through OpenCV's V4L2 ``cap.set`` properties.

    Unlike macOS, the Linux V4L2 backend honors ``CAP_PROP_AUTO_EXPOSURE`` and
    ``CAP_PROP_EXPOSURE``, so no external helper is needed. The exposure scale is
    driver-specific; failures are absorbed by the fail-safe path.
    """

    min_exposure = V4L2_MIN_EXPOSURE
    max_exposure = V4L2_MAX_EXPOSURE
    default_exposure = V4L2_DEFAULT_EXPOSURE

    def __init__(self, camera_index=0, cap=None):
        """Store the capture handle (required for property writes), then probe.

        Args:
            camera_index: Device index (kept for symmetry/logging).
            cap: The already-opened ``cv2.VideoCapture`` used for ``set``/``get``.
        """
        self._cap = cap
        super().__init__(camera_index)

    def _probe(self):
        """Usable only if we hold an opened capture to write properties through."""
        return self._cap is not None and self._cap.isOpened()

    def _enter_manual_mode(self):
        """Request manual auto-exposure mode via V4L2. Return True if accepted."""
        try:
            return bool(self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,
                                      V4L2_AUTO_EXPOSURE_MANUAL))
        except cv2.error:
            return False

    def _write_exposure(self, exposure):
        """Write the exposure property via V4L2. Return True if accepted."""
        try:
            return bool(self._cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure)))
        except cv2.error:
            return False


def make_exposure_controller(camera_index=0, cap=None):
    """Return the appropriate exposure controller for the current platform.

    macOS uses the ``uvc-util-app`` helper; Linux uses OpenCV's V4L2 properties;
    any other platform (or a failed probe) yields a disabled base controller whose
    methods are all no-ops. The returned object always exposes the same interface,
    so callers never branch on the platform themselves.

    Args:
        camera_index: The capture device index.
        cap: The already-opened ``cv2.VideoCapture`` (needed by the Linux backend).

    Returns:
        An ``ExposureController`` instance (possibly disabled, never None).
    """
    system = platform.system()
    if system == "Darwin":
        return UvcUtilExposureController(camera_index)
    if system == "Linux":
        return V4L2ExposureController(camera_index, cap)
    return ExposureController(camera_index)
