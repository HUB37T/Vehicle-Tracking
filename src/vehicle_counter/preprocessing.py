"""Optional intensity, smoothing and sharpening operations on BGR uint8 frames."""

import numpy as np
from numpy.typing import NDArray

from vehicle_counter.contracts import PreprocessConfig


def apply_clahe_luminance(
    luminance: NDArray[np.uint8], config: PreprocessConfig
) -> NDArray[np.uint8]:
    import cv2

    return cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit, tileGridSize=config.clahe_tile_grid_size
    ).apply(luminance)


def preprocess_frame(frame_bgr: NDArray[np.uint8], config: PreprocessConfig) -> NDArray[np.uint8]:
    """Return a new image, preserving the source for original-frame annotation."""
    if (
        not isinstance(frame_bgr, np.ndarray)
        or frame_bgr.dtype != np.uint8
        or frame_bgr.ndim != 3
        or frame_bgr.shape[2] != 3
        or min(frame_bgr.shape[:2]) == 0
    ):
        raise ValueError("Frame harus berupa citra BGR uint8 dengan bentuk (height, width, 3).")
    if config.mode == "none":
        return frame_bgr.copy()
    if config.mode == "gamma":
        levels = np.arange(256) / 255
        lookup = np.rint(255 * levels**config.gamma).astype(np.uint8)
        return lookup[frame_bgr]
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "Mode ini memerlukan paket vision: pip install -e '.[vision]'."
        ) from error
    if config.mode in ("gaussian", "unsharp"):
        kernel = (config.gaussian_kernel_size, config.gaussian_kernel_size)
        blurred = cv2.GaussianBlur(frame_bgr, kernel, sigmaX=0)
        if config.mode == "gaussian":
            return blurred
        amount = config.unsharp_amount
        return cv2.addWeighted(frame_bgr, 1 + amount, blurred, -amount, 0)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = apply_clahe_luminance(lab[:, :, 0], config)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
