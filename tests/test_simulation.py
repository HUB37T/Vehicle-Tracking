import json
import subprocess
import sys

from vehicle_counter.simulation import run_simulation


def test_synthetic_demo_counts_only_two_valid_crossings():
    result = run_simulation()
    assert result["source_kind"] == "synthetic"
    assert result["is_model_inference"] is False
    assert result["summary"]["total"] == len(result["events"]) == 2
    assert {(e["track_id"], e["direction"]) for e in result["events"]} == {
        (1, "A_TO_B"),
        (2, "B_TO_A"),
    }
    assert run_simulation() == result


def test_cli_can_run_outside_repository_and_write_labeled_fixture(tmp_path):
    output = tmp_path / "simulation.json"
    result = subprocess.run(
        [sys.executable, "-m", "vehicle_counter.simulation", "--output", str(output)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    printed = json.loads(result.stdout)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved == printed
    assert saved["source_kind"] == "synthetic"
    assert saved["summary"]["total"] == 2


def test_cli_reports_bad_config_with_nonzero_exit(tmp_path):
    config = tmp_path / "invalid.yaml"
    config.write_text("min_track_hits: 0", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "vehicle_counter.simulation", "--config", str(config)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Konfigurasi gagal dibaca" in result.stderr
