from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from vehicle_counter.contracts import RunConfig, VideoMetadata
from vehicle_counter.tracking import (
    YoloByteTracker,
    load_tracker_settings,
    observations_from_tracks,
    resolve_weights,
)


def test_mapping_uses_matched_raw_detection_and_original_coordinates():
    class Detections(SimpleNamespace):
        def __len__(self):
            return 2

    detections = Detections(
        cls=np.array([10, 20]),
        conf=np.array([0.7, 0.8]),
        xyxy=np.array([[1, 2, 11, 12], [100, 200, 300, 400]]),
    )
    rows = np.array([[105, 205, 305, 405, 7, 0.8, 20, 1]])
    observations = observations_from_tracks(
        rows,
        detections,
        {10: "car", 20: "bus"},
        frame_index=3,
        timestamp_seconds=0.3,
    )
    assert len(observations) == 1
    observation = observations[0]
    assert observation.track_id == 7
    assert observation.class_name == "bus"
    assert observation.bbox_xyxy == (100, 200, 300, 400)
    assert observation.anchor == (200, 400)
    assert (observation.frame_index, observation.timestamp_seconds) == (3, 0.3)


@pytest.mark.parametrize(
    "row",
    [
        [0, 0, 10, 10, 1, 0.9, 0, 5],
        [0, 0, 10, 10, 1.5, 0.9, 0, 0],
        [0, 0, 10, 10, 1, 0.9, 0, float("nan")],
    ],
)
def test_invalid_detection_mapping_fails_instead_of_fabricating_observation(row):
    detections = type("Detections", (), {"__len__": lambda self: 1})()
    with pytest.raises(ValueError):
        observations_from_tracks(
            np.array([row]),
            detections,
            {0: "car"},
            frame_index=0,
            timestamp_seconds=0,
        )


def test_missing_weights_raise_without_implicit_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare-model"):
        resolve_weights(str(tmp_path / "missing.pt"))


def test_tracker_config_requires_bytetrack_and_valid_thresholds(tmp_path):
    path = tmp_path / "tracker.yaml"
    path.write_text("tracker_type: botsort", encoding="utf-8")
    with pytest.raises(ValueError, match="bytetrack"):
        load_tracker_settings(str(path))
    path.write_text("tracker_type: bytetrack", encoding="utf-8")
    with pytest.raises(ValueError, match="lengkap"):
        load_tracker_settings(str(path))


def test_real_bytetrack_association_with_fake_detector_and_fresh_run(monkeypatch, tmp_path):
    ultralytics = pytest.importorskip("ultralytics")
    from ultralytics.engine.results import Boxes

    calls = []

    class FakeDetector:
        task = "detect"
        names = {4: "car", 7: "motorcycle", 9: "bus", 12: "truck"}

        def predict(self, **kwargs):
            calls.append(kwargs)
            x = 20 + int(kwargs["source"][0, 0, 0])
            data = np.array([[x, 20, x + 30, 70, 0.9, 4]], dtype=np.float32)
            return [SimpleNamespace(boxes=Boxes(data, (100, 100)))]

    monkeypatch.setattr(ultralytics, "YOLO", lambda _: FakeDetector())
    weights = tmp_path / "fake-weights.pt"
    weights.write_bytes(b"test fixture, not model weights")
    config = replace(RunConfig(), model_weights=str(weights))
    metadata = VideoMetadata("synthetic.mp4", 100, 100, 10)
    first = YoloByteTracker(config, metadata)
    ids = []
    for index in range(3):
        frame = np.full((100, 100, 3), index, dtype=np.uint8)
        observations = first.track_frame(frame, frame_index=index, timestamp_seconds=index / 10)
        assert observations[0].bbox_xyxy[0] == 20 + index
        ids.append(observations[0].track_id)
    assert len(set(ids)) == 1
    assert calls[0]["classes"] == [4, 7, 9, 12]
    assert calls[0]["save"] is False
    with pytest.raises(ValueError, match="berurutan"):
        first.track_frame(frame, frame_index=0, timestamp_seconds=0)
    second = YoloByteTracker(config, metadata)
    observations = second.track_frame(np.zeros_like(frame), frame_index=0, timestamp_seconds=0)
    assert observations[0].track_id == ids[0]


def test_empty_frames_advance_actual_bytetrack(monkeypatch, tmp_path):
    ultralytics = pytest.importorskip("ultralytics")
    from ultralytics.engine.results import Boxes

    class EmptyDetector:
        task = "detect"
        names = {i: name for i, name in enumerate(RunConfig().vehicle_classes)}

        def predict(self, **kwargs):
            return [SimpleNamespace(boxes=Boxes(np.empty((0, 6)), (100, 100)))]

    monkeypatch.setattr(ultralytics, "YOLO", lambda _: EmptyDetector())
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"fixture")
    adapter = YoloByteTracker(
        replace(RunConfig(), model_weights=str(weights)), VideoMetadata("empty.mp4", 100, 100, 10)
    )
    for index in range(3):
        assert (
            adapter.track_frame(
                np.zeros((100, 100, 3), dtype=np.uint8),
                frame_index=index,
                timestamp_seconds=index / 10,
            )
            == []
        )
    assert adapter._tracker.frame_id == 3
