from dataclasses import replace

import pytest

from vehicle_counter.contracts import CountEvent, CountingLineConfig, TrackObservation
from vehicle_counter.geometry import PixelCountingLine
from vehicle_counter.line_editor import decode_editor
from vehicle_counter.run_analysis import CrossingDiagnostics, timeline_rows


def test_timeline_keeps_zero_intervals_and_exact_boundary_events():
    events = [
        CountEvent("r", "l", i, i * 50, t, "car", "A_TO_B")
        for i, t in enumerate((0.0, 4.99, 5.0, 10.0))
    ]
    rows = timeline_rows(events, duration=10.1)
    assert len(rows) == 24
    assert sum(r["count"] for r in rows) == 4
    assert [
        r["count"] for r in rows if r["class_name"] == "car" and r["direction"] == "A_TO_B"
    ] == [2, 1, 1]


def test_diagnostics_distinguish_no_tracks_one_side_and_large_gap():
    line = PixelCountingLine.from_config(CountingLineConfig(), width=100, height=100)
    diagnostics = CrossingDiagnostics(line, max_gap=0.5)
    assert "Belum ada deteksi" in diagnostics.finish(0)["messages"][0]
    first = TrackObservation(0, 0, 1, "car", 0.9, (10, 10, 30, 30))
    diagnostics.update([first], [])
    assert "kedua sisi" in diagnostics.finish(0)["messages"][0]
    diagnostics.update(
        [replace(first, frame_index=10, timestamp_seconds=1, bbox_xyxy=(10, 60, 30, 80))], []
    )
    info = diagnostics.finish(0)
    assert info["gap_resets"] == 1
    assert info["examples"][0]["previous_frame_index"] == 0
    assert diagnostics.finish(1)["messages"] == []


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"p1": [0.1, 0.3], "p2": [0.1, 0.3], "roi": [0, 0, 1, 1]},
        {"p1": [-0.1, 0.3], "p2": [0.9, 0.3], "roi": [0, 0, 1, 1]},
        {"p1": [0.1, 0.3], "p2": [0.9, 0.3], "roi": [0, 0, 0, 1]},
    ],
)
def test_line_editor_validates_component_payload(value):
    with pytest.raises(ValueError):
        decode_editor(value, CountingLineConfig())


def test_line_editor_preserves_line_identity_and_tolerance():
    original = CountingLineConfig(line_id="road", epsilon_height_fraction=0.02)
    line, roi = decode_editor(
        {"p1": [0.2, 0.7], "p2": [0.8, 0.7], "roi": [0.1, 0.3, 0.9, 0.9]}, original
    )
    assert line.line_id == "road" and line.epsilon_height_fraction == 0.02
    assert line.p1 == (0.2, 0.7) and roi == (0.1, 0.3, 0.9, 0.9)
