"""One-line, two-direction crossing counter independent of detector and UI."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

from vehicle_counter.contracts import (
    DIRECTIONS,
    VEHICLE_CLASSES,
    CountEvent,
    CountSummary,
    Point,
    RunConfig,
    TrackObservation,
    _nonempty,
)
from vehicle_counter.crossing_duplicates import CrossingDuplicates
from vehicle_counter.geometry import PixelCountingLine

TruckReview = Callable[[NDArray[np.uint8], TrackObservation], bool]


@dataclass(slots=True)
class _TrackState:
    last_seen: float | None = None
    last_anchor: Point | None = None
    stable_side: str | None = None
    crossed_segment: bool = False
    hits: int = 0
    class_hits: dict[str, int] = field(default_factory=dict)
    class_confidence: dict[str, float] = field(default_factory=dict)

    def restart_motion(self) -> None:
        self.last_anchor = None
        self.stable_side = None
        self.crossed_segment = False
        self.hits = 0

    def voted_class(self) -> str:
        return min(
            self.class_hits,
            key=lambda name: (-self.class_hits[name], -self.class_confidence[name], name),
        )


class VehicleCounter:
    """Accept one chronological batch per frame and return newly committed events.

    Construct a new counter with a new run_id to restart analysis. Empty batches
    are allowed. Nonempty batches must contain one timestamp/frame and unique IDs.
    Returned summaries are detached snapshots; the event log is read-only.
    """

    def __init__(self, run_id: str, config: RunConfig, *, width: int, height: int) -> None:
        _nonempty(run_id, "run_id")
        self.run_id = run_id
        self.config = config
        self.line = PixelCountingLine.from_config(config.counting_line, width=width, height=height)
        self._duplicates = CrossingDuplicates(self.line) if config.deduplicate_crossings else None
        self._tracks: dict[int, _TrackState] = {}
        self._counted: set[int] = set()
        self._events: list[CountEvent] = []
        self._counts = {name: dict.fromkeys(DIRECTIONS, 0) for name in VEHICLE_CLASSES}
        self._last_frame = -1
        self._last_timestamp = -1.0

    @property
    def events(self) -> tuple[CountEvent, ...]:
        return tuple(self._events)

    def summary(self) -> CountSummary:
        return CountSummary(
            total=len(self._events),
            by_class_direction={name: dict(values) for name, values in self._counts.items()},
        )

    def update(
        self,
        observations_for_frame: Iterable[TrackObservation],
        *,
        frame_bgr: NDArray[np.uint8] | None = None,
        review_truck: TruckReview | None = None,
    ) -> list[CountEvent]:
        observations = list(observations_for_frame)
        if not observations:
            return []
        self._validate_batch(observations)
        first = observations[0]
        self._last_frame = first.frame_index
        self._last_timestamp = first.timestamp_seconds
        if self._duplicates is not None:
            for observation in observations:
                self._duplicates.observe_counted(observation)
        new_events = []
        for observation in observations:
            if observation.class_name not in self.config.vehicle_classes:
                continue
            if observation.track_id in self._counted:
                continue
            event = self._observe(observation)
            if event is not None:
                self._counted.add(event.track_id)
                if (
                    self._duplicates is not None
                    and frame_bgr is not None
                    and self._duplicates.suppress(event, observation, frame_bgr)
                ):
                    continue
                if (
                    event.class_name == "truck"
                    and frame_bgr is not None
                    and review_truck is not None
                    and review_truck(frame_bgr, observation)
                ):
                    event = replace(event, class_name="car")
                self._events.append(event)
                self._counts[event.class_name][event.direction] += 1
                new_events.append(event)
        return new_events

    def _validate_batch(self, observations: list[TrackObservation]) -> None:
        if any(not isinstance(item, TrackObservation) for item in observations):
            raise ValueError("Counter memerlukan TrackObservation untuk setiap observasi.")
        first = observations[0]
        if first.frame_index <= self._last_frame or first.timestamp_seconds <= self._last_timestamp:
            raise ValueError(
                "Frame dan timestamp batch harus meningkat; buat counter baru untuk run baru."
            )
        ids = set()
        for item in observations:
            if (item.frame_index, item.timestamp_seconds) != (
                first.frame_index,
                first.timestamp_seconds,
            ):
                raise ValueError("Satu batch harus berasal dari frame dan timestamp yang sama.")
            if item.track_id in ids:
                raise ValueError("Satu Track ID hanya boleh memiliki satu observasi per frame.")
            ids.add(item.track_id)

    def _observe(self, observation: TrackObservation) -> CountEvent | None:
        state = self._tracks.setdefault(observation.track_id, _TrackState())
        if (
            state.last_seen is not None
            and observation.timestamp_seconds - state.last_seen
            > self.config.max_crossing_gap_seconds + 1e-9
        ):
            state.restart_motion()
        state.hits += 1
        name = observation.class_name
        state.class_hits[name] = state.class_hits.get(name, 0) + 1
        state.class_confidence[name] = (
            state.class_confidence.get(name, 0.0) + observation.confidence
        )
        anchor = observation.anchor
        side = self.line.side(anchor)
        if state.last_anchor is not None:
            state.crossed_segment |= self.line.intersects_path(state.last_anchor, anchor)
        event = None
        if side is not None:
            if (
                state.stable_side is not None
                and side != state.stable_side
                and state.crossed_segment
                and state.hits >= self.config.min_track_hits
            ):
                event = CountEvent(
                    run_id=self.run_id,
                    line_id=self.config.counting_line.line_id,
                    track_id=observation.track_id,
                    frame_index=observation.frame_index,
                    timestamp_seconds=observation.timestamp_seconds,
                    class_name=state.voted_class(),
                    direction=f"{state.stable_side}_TO_{side}",
                )
            state.stable_side = side
            state.crossed_segment = False
        state.last_anchor = anchor
        state.last_seen = observation.timestamp_seconds
        return event
