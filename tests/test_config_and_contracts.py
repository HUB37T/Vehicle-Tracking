import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from vehicle_counter.config import config_to_dict, load_config
from vehicle_counter.contracts import (
    CountingLineConfig,
    PreprocessConfig,
    RunConfig,
    TrackObservation,
    VideoMetadata,
)

ROOT = Path(__file__).resolve().parents[1]


def test_repository_config_matches_contract_defaults():
    config = load_config(ROOT / "configs" / "default.yaml")
    assert config == RunConfig()
    with pytest.raises(FrozenInstanceError):
        config.min_track_hits = 9
    snapshot = config_to_dict(config)
    snapshot["counting_line"]["p1"] = [0, 0]
    assert config.counting_line.p1 == (0.1, 0.6)
    assert json.loads(json.dumps(config_to_dict(config)))["preprocess"]["mode"] == "none"


@pytest.mark.parametrize(
    "content",
    [
        "",
        "- not-a-mapping",
        "detector_confidence: 0.2",
        "preprocess:\n  typo: 1",
        "counting_line:\n  p1: [0.5, 0.5]\n  p2: [0.5, 0.5]",
        "counting_line:\n  p1: [-0.1, 0.5]",
        "detector_conf: .nan",
        "min_track_hits: true",
        "preprocess: null",
        "preprocess: [invalid",
    ],
)
def test_invalid_yaml_config_is_rejected(tmp_path, content):
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"detector_conf": 1.1},
        {"detector_conf": -1},
        {"max_crossing_gap_seconds": 0},
        {"min_track_hits": 0},
        {"imgsz": 0},
        {"vehicle_classes": ("car",)},
        {"vehicle_classes": ("car", "car", "bus", "truck")},
        {"model_weights": " "},
        {"device": None},
    ],
)
def test_invalid_run_parameters_are_rejected(overrides):
    with pytest.raises(ValueError):
        RunConfig(**overrides)


def test_nested_configs_normalize_sequences_to_immutable_tuples():
    line = CountingLineConfig(p1=[0.1, 0.5], p2=[0.9, 0.5])
    preprocess = PreprocessConfig(clahe_tile_grid_size=[4, 4])
    assert line.p1 == (0.1, 0.5)
    assert preprocess.clahe_tile_grid_size == (4, 4)


@pytest.mark.parametrize(
    "overrides",
    [
        {"mode": "invalid"},
        {"gamma": 0},
        {"gaussian_kernel_size": 4},
        {"unsharp_amount": -1},
        {"clahe_clip_limit": float("inf")},
        {"clahe_clip_limit": 0},
        {"clahe_tile_grid_size": (0, 8)},
        {"clahe_tile_grid_size": (8,)},
        {"clahe_tile_grid_size": (1.5, 8)},
    ],
)
def test_invalid_preprocessing_parameters(overrides):
    with pytest.raises(ValueError):
        PreprocessConfig(**overrides)


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), True])
def test_video_metadata_rejects_invalid_fps(fps):
    with pytest.raises(ValueError):
        VideoMetadata("example.mp4", 640, 480, fps)


def test_video_metadata_allows_unknown_frame_count():
    metadata = VideoMetadata("example.mp4", 640, 480, 25)
    assert metadata.frame_count is None
    assert metadata.duration_seconds is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"track_id": None},
        {"track_id": -1},
        {"frame_index": -1},
        {"timestamp_seconds": float("nan")},
        {"confidence": 1.1},
        {"confidence": float("inf")},
        {"bbox_xyxy": (20, 10, 20, 30)},
        {"bbox_xyxy": (10, 10, 30, float("nan"))},
    ],
)
def test_invalid_observation_is_rejected(overrides):
    valid = TrackObservation(0, 0, 1, "car", 0.9, (10, 10, 30, 40))
    with pytest.raises(ValueError):
        replace(valid, **overrides)


def test_anchor_uses_bottom_center_instead_of_box_center():
    observation = TrackObservation(0, 0, 1, "car", 0.9, (10, 10, 30, 40))
    assert observation.anchor == (20, 40)
