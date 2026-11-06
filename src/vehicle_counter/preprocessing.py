"""BGR uint8 preprocessing; none preserves the frame, CLAHE works on LAB L."""

import numpy as np
from numpy.typing import NDArray

from vehicle_counter.contracts import PreprocessConfig


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
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "Mode CLAHE memerlukan paket vision: pip install -e '.[vision]'."
        ) from error
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=config.clahe_tile_grid_size,
    )
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
