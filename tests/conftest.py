from dataclasses import replace

import pytest

from vehicle_counter.contracts import CountingLineConfig, RunConfig, TrackObservation
from vehicle_counter.counting import VehicleCounter


@pytest.fixture
def make_observation():
    def make(
        frame,
        y,
        *,
        x=50,
        track_id=1,
        class_name="car",
        confidence=0.9,
        timestamp=None,
    ):
        return TrackObservation(
            frame_index=frame,
            timestamp_seconds=frame / 10 if timestamp is None else timestamp,
            track_id=track_id,
            class_name=class_name,
            confidence=confidence,
            bbox_xyxy=(x - 2, y - 5, x + 2, y),
        )

    return make


@pytest.fixture
def make_counter():
    def make(*, run_id="test-run", **overrides):
        config = replace(
            RunConfig(),
            counting_line=CountingLineConfig(p1=(0.2, 0.5), p2=(0.8, 0.5)),
            **overrides,
        )
        return VehicleCounter(run_id, config, width=101, height=101)

    return make
