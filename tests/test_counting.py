from dataclasses import replace

import pytest

from vehicle_counter.contracts import DIRECTIONS, VEHICLE_CLASSES, CountingLineConfig
from vehicle_counter.counting import VehicleCounter


def feed(counter, make_observation, ys, **kwargs):
    events = []
    for frame, y in enumerate(ys):
        events.extend(counter.update([make_observation(frame, y, **kwargs)]))
    return events


@pytest.mark.parametrize(
    ("ys", "direction"),
    [([30, 40, 50, 60], "A_TO_B"), ([70, 60, 50, 40], "B_TO_A")],
)
def test_crossing_has_correct_direction_and_time(make_counter, make_observation, ys, direction):
    counter = make_counter()
    events = feed(counter, make_observation, ys)
    assert len(events) == 1
    event = events[0]
    assert (event.run_id, event.line_id, event.track_id) == ("test-run", "main", 1)
    assert (event.frame_index, event.timestamp_seconds) == (3, 0.3)
    assert event.direction == direction
    assert counter.summary().by_class_direction["car"][direction] == 1


@pytest.mark.parametrize(
    "ys",
    [
        [30, 30, 30, 30],
        [30, 40, 50, 40, 30],
        [49.5, 50.5, 50, 49.8, 50.8],
        [50, 50, 60, 70],
        [30, 40, 49.5, 50.5, 49.8, 40],
    ],
)
def test_stationary_neutral_touch_and_initial_neutral_do_not_count(
    make_counter, make_observation, ys
):
    counter = make_counter()
    assert feed(counter, make_observation, ys) == []
    assert counter.summary().total == 0


def test_one_id_only_counts_once_even_after_uturn_and_long_gap(make_counter, make_observation):
    counter = make_counter()
    events = feed(counter, make_observation, [30, 40, 60, 50, 30, 60, 30])
    assert len(events) == 1
    counter.update([make_observation(100, 70)])
    counter.update([make_observation(101, 30)])
    assert counter.summary().total == 1


@pytest.mark.parametrize("x", [10, 90])
def test_crossing_extension_outside_segment_does_not_count(make_counter, make_observation, x):
    assert feed(make_counter(), make_observation, [30, 40, 60], x=x) == []


@pytest.mark.parametrize("x", [20, 80])
def test_line_endpoints_are_included(make_counter, make_observation, x):
    assert len(feed(make_counter(), make_observation, [30, 40, 60], x=x)) == 1


def test_neutral_path_uses_actual_segments_instead_of_straight_shortcut(
    make_counter, make_observation
):
    counter = make_counter()
    points = [(50, 30), (10, 49.5), (10, 50.5), (50, 70)]
    events = []
    for frame, (x, y) in enumerate(points):
        events.extend(counter.update([make_observation(frame, y, x=x)]))
    assert events == []


def test_crossing_flag_survives_neutral_frames(make_counter, make_observation):
    counter = make_counter()
    assert len(feed(counter, make_observation, [30, 49.5, 50.5, 50, 70])) == 1


def test_insufficient_hits_do_not_retroactively_count(make_counter, make_observation):
    counter = make_counter(min_track_hits=4)
    assert feed(counter, make_observation, [30, 60, 70, 80, 90]) == []


def test_long_observation_gap_restarts_motion_history(make_counter, make_observation):
    counter = make_counter()
    counter.update([make_observation(0, 30)])
    counter.update([make_observation(1, 40)])
    assert counter.update([]) == []
    assert counter.update([make_observation(20, 70)]) == []
    assert counter.update([make_observation(21, 60)]) == []
    events = counter.update([make_observation(22, 30)])
    assert len(events) == 1
    assert events[0].direction == "B_TO_A"


def test_gap_exactly_at_limit_is_allowed(make_counter, make_observation):
    counter = make_counter()
    counter.update([make_observation(0, 30, timestamp=0)])
    counter.update([make_observation(1, 40, timestamp=0.5)])
    assert len(counter.update([make_observation(2, 70, timestamp=1.0)])) == 1


@pytest.mark.parametrize(
    ("classes", "confidences", "winner"),
    [
        (["car", "car", "truck"], [0.3, 0.3, 1.0], "car"),
        (["car", "bus", "truck"], [0.5, 0.9, 0.4], "bus"),
        (["truck", "car", "bus"], [0.8, 0.8, 0.8], "bus"),
    ],
)
def test_class_vote_uses_frequency_then_confidence_then_alphabet(
    make_counter, make_observation, classes, confidences, winner
):
    counter = make_counter()
    for frame, y in enumerate([30, 40, 70]):
        counter.update(
            [make_observation(frame, y, class_name=classes[frame], confidence=confidences[frame])]
        )
    assert counter.events[0].class_name == winner
    counter.update([make_observation(3, 80, class_name="motorcycle")])
    assert counter.events[0].class_name == winner


def test_all_classes_and_directions_aggregate_from_event_log(make_counter, make_observation):
    counter = make_counter()
    for frame in range(3):
        observations = []
        for index, name in enumerate(VEHICLE_CLASSES):
            observations.append(
                make_observation(frame, [30, 40, 70][frame], track_id=index, class_name=name)
            )
            observations.append(
                make_observation(frame, [70, 60, 30][frame], track_id=index + 4, class_name=name)
            )
        counter.update(observations)
    summary = counter.summary()
    assert summary.total == len(counter.events) == 8
    assert all(
        summary.by_class_direction[name][d] == 1 for name in VEHICLE_CLASSES for d in DIRECTIONS
    )
    for name in VEHICLE_CLASSES:
        for direction in DIRECTIONS:
            assert summary.by_class_direction[name][direction] == sum(
                event.class_name == name and event.direction == direction
                for event in counter.events
            )


def test_unsupported_class_is_ignored(make_counter, make_observation):
    assert feed(make_counter(), make_observation, [30, 40, 70], class_name="person") == []


def test_empty_summary_has_eight_zero_buckets_and_is_detached(make_counter):
    counter = make_counter()
    first = counter.summary()
    assert first.total == 0
    assert len(first.by_class_direction) == 4
    assert sum(len(values) for values in first.by_class_direction.values()) == 8
    first.by_class_direction["car"]["A_TO_B"] = 99
    assert counter.summary().by_class_direction["car"]["A_TO_B"] == 0


def test_new_run_is_independent(make_counter, make_observation):
    first = make_counter()
    feed(first, make_observation, [30, 40, 70])
    second = make_counter(run_id="new-run")
    assert second.summary().total == 0
    assert len(feed(second, make_observation, [30, 40, 70])) == 1
    assert second.events[0].run_id == "new-run"


@pytest.mark.parametrize("kind", ["duplicate", "mixed_frame", "mixed_timestamp"])
def test_invalid_batch_is_rejected_before_state_changes(make_counter, make_observation, kind):
    counter = make_counter()
    first = make_observation(0, 30)
    second = {
        "duplicate": make_observation(0, 30),
        "mixed_frame": make_observation(1, 30, track_id=2),
        "mixed_timestamp": make_observation(0, 30, track_id=2, timestamp=1),
    }[kind]
    with pytest.raises(ValueError):
        counter.update([first, second])
    assert counter.update([first]) == []
    assert counter.summary().total == 0


def test_replayed_or_out_of_order_frame_is_rejected(make_counter, make_observation):
    counter = make_counter()
    counter.update([make_observation(1, 30)])
    with pytest.raises(ValueError, match="meningkat"):
        counter.update([make_observation(1, 40)])
    with pytest.raises(ValueError, match="meningkat"):
        counter.update([make_observation(2, 40, timestamp=0.05)])


def test_reversing_line_reverses_direction(make_counter, make_observation):
    original = make_counter()
    line = CountingLineConfig(p1=(0.8, 0.5), p2=(0.2, 0.5))
    config = replace(original.config, counting_line=line)
    counter = VehicleCounter("reversed", config, width=101, height=101)
    assert feed(counter, make_observation, [30, 40, 70])[0].direction == "B_TO_A"
