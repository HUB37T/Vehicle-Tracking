import json
from dataclasses import replace

import pytest

from vehicle_counter.camera_presets import CameraPreset, load_camera_presets, save_camera_preset


@pytest.fixture
def preset():
    return CameraPreset("Simpang A – Kamera 1", (0.85, 0.64), (0.15, 0.62), (0.21, 0.32, 0.9, 0.87))


def test_preset_json_round_trip_preserves_direction_and_area(preset):
    assert CameraPreset.from_json(preset.to_json().encode("utf-8-sig")) == preset


@pytest.mark.parametrize(
    "change",
    [
        {"version": True},
        {"version": 2},
        {"name": "\n"},
        {"p1": [0.15, 0.62]},
        {"p2": "0.15, 0.62"},
        {"roi": [0.9, 0.3, 0.2, 0.8]},
        {"roi": [False, 0.3, 1, 1]},
        {"roi": [0, 0.3, float("nan"), 1]},
        {"video_path": "/untrusted.mp4"},
    ],
)
def test_rejects_invalid_preset(preset, change):
    with pytest.raises(ValueError):
        CameraPreset.from_json(json.dumps({**json.loads(preset.to_json()), **change}).encode())


def test_local_library_preserves_existing_preset_on_name_conflict(tmp_path, preset):
    path = tmp_path / "outputs" / "camera-presets.sqlite3"
    assert not load_camera_presets(path)
    save_camera_preset(path, preset)
    save_camera_preset(path, preset)
    with pytest.raises(ValueError, match="Nama preset sudah dipakai"):
        save_camera_preset(path, replace(preset, name=preset.name.upper(), p1=(0.71, 0.64)))
    assert load_camera_presets(path) == {preset.key: preset}
