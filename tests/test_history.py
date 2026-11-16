import json
import shutil
from pathlib import Path

import pytest

from vehicle_counter.history import list_runs, load_run


def make_run(root, name="run-1", *, status="completed"):
    directory = root / name
    directory.mkdir(parents=True)
    artifacts = {}
    for key, name in (
        ("events_csv", "events.csv"),
        ("counts_csv", "counts.csv"),
        ("annotated_video", "annotated.mp4"),
    ):
        path = directory / name
        path.write_text("fixture, not real inference", encoding="utf-8")
        artifacts[key] = str(path.resolve())
    info = {
        "run_id": directory.name,
        "status": status,
        "started_at_utc": "2026-11-15T00:00:00Z",
        "video_metadata": {"source_name": "fixture.mp4"},
        "config": {"model_weights": "yolo11n.pt", "preprocess": {"mode": "none"}},
        "summary": {"total": 0, "by_class_direction": {}},
        "artifact_paths": artifacts,
    }
    path = directory / "run.json"
    path.write_text(json.dumps(info), encoding="utf-8")
    return path


def test_history_lists_valid_runs_and_reopens_after_move(tmp_path):
    original = tmp_path / "original"
    path = make_run(original)
    broken = original / "broken"
    broken.mkdir()
    (broken / "run.json").write_text("{", encoding="utf-8")
    assert [r["run_id"] for r in list_runs(original)] == ["run-1"]
    relocated = tmp_path / "moved"
    shutil.copytree(path.parent, relocated)
    result, info = load_run(relocated / "run.json")
    assert result.run_id == "run-1" and result.summary.total == info["summary"]["total"]
    assert all(Path(value).parent == relocated for value in result.artifact_paths.values())


def test_running_and_missing_artifacts_are_explicit_errors(tmp_path):
    path = make_run(tmp_path, status="running")
    with pytest.raises(ValueError, match="berlangsung"):
        load_run(path)
    info = json.loads(path.read_text())
    info["status"] = "completed"
    path.write_text(json.dumps(info))
    (path.parent / "annotated.mp4").unlink()
    with pytest.raises(ValueError, match="tidak lengkap"):
        load_run(path)


def test_history_does_not_open_artifacts_outside_run(tmp_path):
    path = make_run(tmp_path)
    info = json.loads(path.read_text())
    info["artifact_paths"]["escape"] = str(tmp_path / "secret.json")
    (tmp_path / "secret.json").write_text("fixture")
    path.write_text(json.dumps(info))
    result, _ = load_run(path)
    assert "escape" not in result.artifact_paths
