import numpy as np
import pytest

from vehicle_counter.contracts import PreprocessConfig
from vehicle_counter.geometry import PixelCountingLine
from vehicle_counter.preprocessing import preprocess_frame


@pytest.mark.parametrize(
    ("mode", "roi", "expected_stages"),
    [
        ("none", None, ()),
        ("gamma", None, ("Gamma correction",)),
        ("gaussian", (10, 5, 50, 40), ("Gaussian smoothing", "Input detektor")),
        ("unsharp", None, ("Unsharp masking",)),
        ("median", None, ("Filter median",)),
        ("clahe", None, ("Ruang warna | BGR ke Lab", "CLAHE | L sesudah", "Input detektor")),
        (
            "bilateral_clahe",
            None,
            ("Ruang warna | BGR ke Lab", "Bilateral + CLAHE | L sesudah", "Input detektor"),
        ),
    ],
)
def test_evidence_shows_active_stages_and_processed_frame(monkeypatch, mode, roi, expected_stages):
    cv2 = pytest.importorskip("cv2")
    from vehicle_counter.run_evidence import render_evidence
    from vehicle_counter.visualization import FrameAnnotator

    titles = []
    thumbnails = []
    resize = cv2.resize

    def capture_title(canvas, label, *args):
        if label[:2].isdigit():
            titles.append(label[4:])

    def capture_thumbnail(image, *args, **kwargs):
        thumbnails.append(image)
        return resize(image, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture_title)
    monkeypatch.setattr(cv2, "resize", capture_thumbnail)
    source = np.full((48, 64, 3), 40, dtype=np.uint8)
    source[:, 32:] = (95, 150, 200)
    config = PreprocessConfig(mode=mode)
    model_input = preprocess_frame(source, config)
    render_evidence(
        source,
        model_input,
        [],
        [],
        FrameAnnotator(PixelCountingLine((0, 24), (63, 24), 1)),
        config,
        frame_index=10,
        timestamp_seconds=1.0,
        detection_roi_pixels=roi,
    )

    assert titles == ["Frame sumber", *expected_stages, "Deteksi YOLO", "Tracking ByteTrack"]
    np.testing.assert_array_equal(thumbnails[0], source)
    if mode in ("gamma", "gaussian", "unsharp", "median"):
        np.testing.assert_array_equal(thumbnails[1], model_input)
    if roi:
        x1, y1, x2, y2 = roi
        np.testing.assert_array_equal(thumbnails[2], model_input[y1:y2, x1:x2])
