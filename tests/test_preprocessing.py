import numpy as np
import pytest

from vehicle_counter.contracts import PreprocessConfig
from vehicle_counter.preprocessing import preprocess_frame


def test_none_preserves_bgr_pixels_without_aliasing_source():
    frame = np.array([[[255, 10, 0], [0, 10, 255]]], dtype=np.uint8)
    output = preprocess_frame(frame, PreprocessConfig())
    np.testing.assert_array_equal(output, frame)
    output[0, 0] = 0
    np.testing.assert_array_equal(frame[0, 0], [255, 10, 0])


@pytest.mark.parametrize(
    "frame",
    [
        None,
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 4), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
        np.zeros((0, 10, 3), dtype=np.uint8),
    ],
)
def test_invalid_input_frame_is_rejected(frame):
    with pytest.raises(ValueError, match="BGR uint8"):
        preprocess_frame(frame, PreprocessConfig())


def test_clahe_changes_dark_gradient_without_mutating_original():
    pytest.importorskip("cv2")
    ramp = np.tile(np.linspace(20, 70, 64, dtype=np.uint8), (64, 1))
    frame = np.stack([ramp, ramp, ramp], axis=-1)
    original = frame.copy()
    output = preprocess_frame(frame, PreprocessConfig(mode="clahe"))
    assert output.shape == frame.shape
    assert output.dtype == np.uint8
    assert not np.array_equal(output, original)
    np.testing.assert_array_equal(frame, original)


def test_gamma_below_one_brightens_midtones_and_preserves_endpoints():
    frame = np.array([[[0, 64, 255]]], dtype=np.uint8)
    result = preprocess_frame(frame, PreprocessConfig(mode="gamma", gamma=0.5))
    np.testing.assert_array_equal(result, [[[0, 128, 255]]])


def test_gaussian_and_unsharp_move_edge_contrast_in_opposite_directions():
    pytest.importorskip("cv2")
    frame = np.full((11, 12, 3), 80, dtype=np.uint8)
    frame[:, 6:] = 160
    blurred = preprocess_frame(frame, PreprocessConfig(mode="gaussian"))
    sharpened = preprocess_frame(frame, PreprocessConfig(mode="unsharp"))
    assert sharpened[5, 5, 0] < frame[5, 5, 0] < blurred[5, 5, 0]
    assert blurred[5, 6, 0] < frame[5, 6, 0] < sharpened[5, 6, 0]
