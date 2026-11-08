from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from vehicle_counter.tracking import load_tracker_settings, observations_from_tracks


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_age", 0),
        ("n_init", True),
        ("nn_budget", 1.5),
        ("max_cosine_distance", float("nan")),
        ("max_iou_distance", 1.1),
        ("fuse_score", True),
    ],
)
def test_deepsort_rejects_invalid_or_unrelated_settings(tmp_path, key, value):
    _, settings = load_tracker_settings("configs/deepsort.yaml")
    settings[key] = value
    path = tmp_path / "tracker.yaml"
    path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    with pytest.raises(ValueError):
        load_tracker_settings(str(path))


@pytest.mark.parametrize("portable", [False, True])
def test_real_deepsort_uses_current_detection_and_does_not_emit_predictions(monkeypatch, portable):
    pytest.importorskip("deep_sort_realtime")
    pytest.importorskip("torchvision")
    from vehicle_counter.deepsort import DeepSortAssociation

    if portable:
        import torch

        monkeypatch.setattr(torch.backends.cpu, "get_cpu_capability", lambda: "DEFAULT")

    class Detections(SimpleNamespace):
        def __len__(self):
            return len(self.cls)

    _, settings = load_tracker_settings("configs/deepsort.yaml")
    tracker = DeepSortAssociation(settings)
    frame = np.random.default_rng(42).integers(0, 256, (100, 100, 3), dtype=np.uint8)
    ids = []
    for index in range(5):
        box = [20 + index, 20, 50 + index, 75]
        detections = Detections(
            xyxy=np.array([box]),
            conf=np.array([0.9]),
            cls=np.array([2]),
        )
        rows = tracker.update(detections, img=frame)
        if index < settings["n_init"] - 1:
            assert rows.shape == (0, 8)
            continue
        observations = observations_from_tracks(
            rows,
            detections,
            {2: "car"},
            frame_index=index,
            timestamp_seconds=index / 10,
        )
        assert len(observations) == 1
        assert observations[0].bbox_xyxy == tuple(box)
        ids.append(observations[0].track_id)
    assert len(set(ids)) == 1
    empty = Detections(xyxy=np.empty((0, 4)), conf=np.array([]), cls=np.array([]))
    assert tracker.update(empty, img=frame).shape == (0, 8)
    assert len(tracker._tracker.tracker.tracks) == 1  # retained Kalman prediction, never counted
    assert tracker.metadata["vehicle_reid_finetuned"] is False
    assert len(tracker.metadata["weights_sha256"]) == 64
