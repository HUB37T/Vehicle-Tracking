from dataclasses import replace

import pytest

from vehicle_counter.contracts import CongestionConfig, TrackObservation
from vehicle_counter.traffic import TrafficMonitor


def monitor(**overrides):
    config = replace(CongestionConfig(enabled=True, roi=(0, 0, 1, 1)), **overrides)
    return TrafficMonitor(config, width=400, height=180, max_gap=0.5)


def tracks(frame, *, dx=0, count=3, id_offset=0):
    return [
        TrackObservation(
            frame, frame / 10, i + id_offset, "car", 0.9, (i * 110 + dx, 30, i * 110 + 70 + dx, 120)
        )
        for i in range(count)
    ]


def test_dense_stationary_queue_requires_motion_history_and_sustained_evidence():
    traffic = monitor()
    for frame in range(40):
        traffic.update(tracks(frame), frame_index=frame, timestamp=frame / 10)
    assert all(s.status != "congested" for s in traffic.samples)
    traffic.update(tracks(40), frame_index=40, timestamp=4)
    assert traffic.current.status == "congested"
    assert traffic.current.slow_fraction == 1
    assert traffic.current.motion_track_count == 3
    assert traffic.current.median_speed_box_lengths_per_second == 0


def test_dense_moving_traffic_and_single_parked_car_do_not_trigger_congestion():
    for count, speed in ((3, 20), (1, 0)):
        traffic = monitor()
        for frame in range(50):
            traffic.update(
                tracks(frame, dx=frame / 10 * speed, count=count),
                frame_index=frame,
                timestamp=frame / 10,
            )
        assert all(s.status != "congested" for s in traffic.samples)


def test_id_churn_and_missing_observations_cannot_supply_stationary_evidence():
    traffic = monitor()
    for frame in range(60):
        traffic.update(tracks(frame, id_offset=frame * 3), frame_index=frame, timestamp=frame / 10)
    assert traffic.current.status == "insufficient_data"
    assert all(s.status != "congested" for s in traffic.samples)
    traffic.update([], frame_index=66, timestamp=6.6)
    assert traffic.current.status == "empty"


def test_long_gap_clears_motion_and_confirmed_congestion():
    traffic = monitor()
    for frame in range(46):
        traffic.update(tracks(frame), frame_index=frame, timestamp=frame / 10)
    assert traffic.current.status == "congested"
    traffic.update(tracks(60), frame_index=60, timestamp=6)
    assert traffic.current.status == "insufficient_data"


def test_occupancy_is_union_of_boxes_and_roi_excludes_outside_anchors():
    traffic = monitor()
    identical = [replace(o, bbox_xyxy=(10, 30, 80, 120)) for o in tracks(0)]
    traffic.update(identical, frame_index=0, timestamp=0)
    one = monitor()
    one.update(identical[:1], frame_index=0, timestamp=0)
    assert traffic.current.occupancy_fraction == one.current.occupancy_fraction
    assert traffic.current.vehicle_count == 1
    outside = monitor(roi=(0, 0, 0.2, 0.2))
    outside.update(tracks(0), frame_index=0, timestamp=0)
    assert outside.current.vehicle_count == 0


def test_recovery_requires_duration_of_clear_motion():
    traffic = monitor(motion_window_seconds=1, min_motion_seconds=0.5, hold_seconds=1)
    for frame in range(26):
        traffic.update(tracks(frame), frame_index=frame, timestamp=frame / 10)
    assert traffic.current.status == "congested"
    for frame in range(26, 36):
        traffic.update(tracks(frame, dx=(frame - 25) * 4), frame_index=frame, timestamp=frame / 10)
    assert traffic.current.status == "congested"
    for frame in range(36, 66):
        traffic.update(tracks(frame, dx=(frame - 25) * 4), frame_index=frame, timestamp=frame / 10)
    assert traffic.current.status != "congested"


def test_dropout_grace_retries_only_a_confirmed_queue_sample():
    traffic = monitor(dropout_grace_seconds=0.2)
    for frame in range(51):
        traffic.update(
            [] if frame == 45 else tracks(frame), frame_index=frame, timestamp=frame / 10
        )
    assert [(s.timestamp_seconds, s.status) for s in traffic.samples[-2:]] == [
        (4, "congested"),
        (4.6, "congested"),
    ]
    for frame in range(51, 60):
        traffic.update([], frame_index=frame, timestamp=frame / 10)
    assert traffic.current.status == "empty"
    first_empty = next(sample for sample in traffic.samples if sample.status == "empty")
    assert first_empty.timestamp_seconds == pytest.approx(5.3)
    assert traffic.current.vehicle_count == 0


def test_congestion_episodes_split_on_recovery_and_include_video_tail():
    traffic = monitor()
    for frame in range(110):
        traffic.update(
            [] if 50 <= frame < 60 else tracks(frame), frame_index=frame, timestamp=frame / 10
        )
    summary = traffic.summary(11)
    assert sum(summary["durations_seconds"].values()) == pytest.approx(11)
    assert summary["durations_seconds"]["congested"] == pytest.approx(2)
    episodes = summary["episodes"]
    assert len(episodes) == 2
    assert [e["start_seconds"] for e in episodes] == [4, 10]
    assert [e["duration_seconds"] for e in episodes] == [1, 1]
    assert [e["ongoing_at_end"] for e in episodes] == [False, True]
    assert sum(e["duration_seconds"] for e in episodes) == summary["durations_seconds"]["congested"]
    with pytest.raises(ValueError, match="lebih pendek"):
        traffic.summary(1)


@pytest.mark.parametrize(
    "changes",
    [
        {"roi": (0, 0.5, 0, 1)},
        {"roi": (0, 0, 1.2, 1)},
        {"min_vehicles": 1},
        {"hold_seconds": float("nan")},
        {"sample_seconds": 0},
        {"min_slow_fraction": 2},
        {"min_motion_seconds": 3},
        {"enabled": 1},
    ],
)
def test_invalid_traffic_configuration(changes):
    with pytest.raises(ValueError):
        replace(CongestionConfig(), **changes)
