import csv
import json
from dataclasses import replace

import pytest

from vehicle_counter.contracts import CountEvent, CountSummary
from vehicle_counter.exporting import export_counts, write_json
from vehicle_counter.simulation import run_simulation


def fixture():
    data = run_simulation()
    return [CountEvent(**event) for event in data["events"]], CountSummary(**data["summary"])


def test_csv_export_checks_derived_totals_and_duplicates_before_writing(tmp_path):
    events, summary = fixture()
    for invalid_events, invalid_summary in (
        (events, replace(summary, total=9)),
        ([*events, events[0]], summary),
        ([replace(events[0], run_id="wrong")], summary),
    ):
        with pytest.raises(ValueError):
            export_counts(
                tmp_path,
                run_id="synthetic-demo",
                line_id="main",
                events=invalid_events,
                summary=invalid_summary,
            )
    assert not (tmp_path / "events.csv").exists()


def test_exported_rows_follow_shared_contract(tmp_path):
    events, summary = fixture()
    paths = export_counts(
        tmp_path, run_id="synthetic-demo", line_id="main", events=iter(events), summary=summary
    )
    with open(paths["events_csv"], newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert set(rows[0]) == {
        "run_id",
        "line_id",
        "track_id",
        "frame_index",
        "timestamp_seconds",
        "class_name",
        "direction",
    }
    with open(paths["counts_csv"], newline="", encoding="utf-8") as handle:
        counts = list(csv.DictReader(handle))
    assert len(counts) == 8
    assert sum(int(row["count"]) for row in counts) == 2


def test_json_metadata_is_atomic_and_rejects_nan(tmp_path):
    path = tmp_path / "run.json"
    write_json(path, {"status": "running"})
    with pytest.raises(ValueError):
        write_json(path, {"status": "completed", "fps": float("nan")})
    assert json.loads(path.read_text()) == {"status": "running"}
    write_json(path, {"status": "completed", "fps": 10.0})
    assert json.loads(path.read_text())["status"] == "completed"
    assert not path.with_suffix(".json.tmp").exists()
