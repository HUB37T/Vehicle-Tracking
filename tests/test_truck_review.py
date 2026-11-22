"""Score fixtures reproduce the gate failure; they are not accuracy measurements."""

import pytest

pytest.importorskip("cv2")

from vehicle_counter.config import config_from_dict, config_to_dict  # noqa: E402
from vehicle_counter.contracts import RunConfig  # noqa: E402
from vehicle_counter.truck_review import light_vehicle_decision  # noqa: E402


@pytest.mark.parametrize(
    "label,score,expected",
    [
        ("minivan", 0.3157, True),
        ("minivan", 0.3875, True),
        ("minivan", 0.29, False),
        ("ambulance", 0.35, False),  # low-confidence commercial lookalike
        ("moving van", 0.8, False),
        ("minibus", 0.8, False),
        ("trailer truck", 0.8, False),
        ("beach wagon", 0.6, True),
    ],
)
def test_relaxed_gate_applies_only_to_minivan(label, score, expected):
    assert light_vehicle_decision(label, score, 0.30) is expected


def test_old_gate_and_stricter_gate_still_work():
    assert not light_vehicle_decision("minivan", 0.3157, 0.40)
    assert light_vehicle_decision("minivan", 0.6, 0.40)
    assert not light_vehicle_decision("minivan", 0.6, 0.70)


def test_minivan_policy_rejects_strong_non_mpv_lookalikes():
    assert light_vehicle_decision("minivan", 0.3157, 0.30, minivan_only=True)
    assert not light_vehicle_decision("racer", 0.5046, 0.30, minivan_only=True)
    assert not light_vehicle_decision("ambulance", 0.6, 0.30, minivan_only=True)
    assert light_vehicle_decision("racer", 0.5046, 0.40)  # legacy policy
    with pytest.raises(ValueError, match="truck_review_minivan_only"):
        RunConfig(truck_review_minivan_only=1)


@pytest.mark.parametrize("threshold", [True, 0, -0.3, 1.1, float("nan"), float("inf")])
def test_review_threshold_rejects_invalid_config(threshold):
    with pytest.raises(ValueError, match="truck_review_light_threshold"):
        RunConfig(truck_review_light_threshold=threshold)


def test_threshold_round_trip_and_old_default():
    assert RunConfig().truck_review_light_threshold == 0.40
    config = RunConfig(truck_review_light_threshold=0.30)
    assert config_from_dict(config_to_dict(config)) == config
    with pytest.raises(ValueError, match="detector_deduplicate_classes"):
        RunConfig(detector_deduplicate_classes=1)


def test_video_label_identifies_review_and_does_not_reuse_yolo_confidence(monkeypatch):
    import cv2
    import numpy as np

    from vehicle_counter.contracts import TrackObservation
    from vehicle_counter.counting import VehicleCounter
    from vehicle_counter.visualization import FrameAnnotator

    counter = VehicleCounter("fixture", RunConfig(), width=320, height=240)
    annotator = FrameAnnotator(counter.line)
    texts = []
    monkeypatch.setattr(cv2, "putText", lambda frame, text, *args: texts.append(text))
    observation = TrackObservation(0, 0, 7, "truck", 0.9, (140, 120, 180, 180))
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    annotator.render(
        frame, [observation], counter.summary(), timestamp_seconds=0, reviewed_track_ids={7}
    )
    assert "car (review) #7" in texts
    assert not any("#7 0.90" in text for text in texts)
    assert observation.class_name == "truck"  # raw detector evidence is preserved
