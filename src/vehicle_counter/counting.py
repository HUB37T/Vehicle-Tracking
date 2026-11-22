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
from vehicle_counter.dense_crossings import DenseCrossingGuard
from vehicle_counter.geometry import PixelCountingLine
from vehicle_counter.polygons import polygon_contains

TruckReview = Callable[[NDArray[np.uint8], TrackObservation], bool]


def _box_area(observation: TrackObservation) -> float:
    x1, y1, x2, y2 = observation.bbox_xyxy
    return (x2 - x1) * (y2 - y1)


@dataclass(slots=True)
class _TrackState:
    last_seen: float | None = None
    last_anchor: Point | None = None
    stable_side: str | None = None
    crossing_point: Point | None = None
    hits: int = 0
    class_hits: dict[str, int] = field(default_factory=dict)
    class_confidence: dict[str, float] = field(default_factory=dict)

    def restart_motion(self) -> None:
        self.last_anchor = None
        self.stable_side = None
        self.crossing_point = None
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
        self._frame_span = (width - 1, height - 1)
        self._duplicates = CrossingDuplicates(self.line) if config.deduplicate_crossings else None
        self._dense_guard = DenseCrossingGuard() if config.guard_dense_crossings else None
        self.dense_crossing_adjustments: list[dict] = []
        self._tracks: dict[int, _TrackState] = {}
        self._counted: set[int] = set()
        self._crossings: list[tuple[CountEvent, Point]] = []
        self._counts = {name: dict.fromkeys(DIRECTIONS, 0) for name in VEHICLE_CLASSES}
        self._last_frame = -1
        self._last_timestamp = -1.0

    @property
    def events(self) -> tuple[CountEvent, ...]:
        return tuple(event for event, _ in self._crossings)

    def summary(self) -> CountSummary:
        return CountSummary(
            total=len(self._crossings),
            by_class_direction={name: dict(values) for name, values in self._counts.items()},
        )

    def lane_counts(self) -> list[dict[str, str | int | None]]:
        """Attribute each committed crossing to the first matching zone."""
        zones = self.config.traffic_zones
        names: dict[str | None, str] = {zone.zone_id: zone.name for zone in zones}
        names[None] = "Di luar zona"
        counts = {zone_id: dict.fromkeys(VEHICLE_CLASSES, 0) for zone_id in names}
        for event, point in self._crossings:
            zone_id = next(
                (zone.zone_id for zone in zones if polygon_contains(zone.polygon, point)), None
            )
            counts[zone_id][event.class_name] += 1
        rows = []
        for zone_id, name in names.items():
            bucket = counts[zone_id]
            rows.append(
                {"zone_id": zone_id, "zone_name": name, **bucket, "total": sum(bucket.values())}
            )
        return rows

    def update(
        self,
        observations_for_frame: Iterable[TrackObservation],
        *,
        frame_bgr: NDArray[np.uint8] | None = None,
        review_truck: TruckReview | None = None,
        review_passenger_conflict: TruckReview | None = None,
    ) -> list[CountEvent]:
        observations = list(observations_for_frame)
        if not observations:
            return []
        self._validate_batch(observations)
        first = observations[0]
        self._last_frame = first.frame_index
        self._last_timestamp = first.timestamp_seconds
        if self._dense_guard is not None:
            self._dense_guard.observe(observations, first.timestamp_seconds)
        if self._duplicates is not None:
            for observation in observations:
                self._duplicates.observe_counted(observation)
        candidates = []
        for observation in observations:
            if observation.class_name not in self.config.vehicle_classes:
                continue
            if observation.track_id in self._counted:
                continue
            crossing = self._observe(observation)
            if crossing is not None:
                event, point = crossing
                self._counted.add(event.track_id)
                candidates.append((event, point, observation))
        if self._dense_guard is not None:
            # For simultaneous nested boxes, process the more complete box first.
            candidates.sort(key=lambda item: -_box_area(item[2]))
        new_events = []
        for event, point, observation in candidates:
            voted_class = event.class_name
            if self._dense_guard is not None:
                event = replace(
                    event, class_name=self._dense_guard.recent_class(event, observation)
                )
                duplicate_id = self._dense_guard.duplicate_of(event)
                if duplicate_id is not None:
                    self._record_dense_adjustment(event, voted_class, duplicate_id)
                    continue
            pre_review_event = event
            if (
                self._duplicates is not None
                and frame_bgr is not None
                and self._duplicates.suppress(event, observation, frame_bgr)
            ):
                continue
            reviewer = review_truck
            if voted_class == "car" and review_passenger_conflict is not None:
                reviewer = review_passenger_conflict
            if (
                event.class_name == "truck"
                and frame_bgr is not None
                and reviewer is not None
                and reviewer(frame_bgr, observation)
            ):
                event = replace(event, class_name="car")
            if self._dense_guard is not None:
                if voted_class != pre_review_event.class_name and voted_class != event.class_name:
                    self._record_dense_adjustment(event, voted_class)
                # Keep the pre-review class for comparison with the next candidate.
                self._dense_guard.accept(pre_review_event)
            x, y = point
            width_span, height_span = self._frame_span
            self._crossings.append((event, (x / width_span, y / height_span)))
            self._counts[event.class_name][event.direction] += 1
            new_events.append(event)
        return new_events

    def _record_dense_adjustment(
        self, event: CountEvent, voted_class: str, kept_track_id: int | None = None
    ) -> None:
        self.dense_crossing_adjustments.append(
            {
                "frame_index": event.frame_index,
                "timestamp_seconds": event.timestamp_seconds,
                "track_id": event.track_id,
                "direction": event.direction,
                "voted_class": voted_class,
                "class_name": event.class_name,
                "decision": "duplicate_suppressed"
                if kept_track_id is not None
                else "class_corrected",
                "kept_track_id": kept_track_id,
            }
        )

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

    def _observe(self, observation: TrackObservation) -> tuple[CountEvent, Point] | None:
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
            point = self.line.path_intersection(state.last_anchor, anchor)
            if point is not None:
                state.crossing_point = point
        crossing = None
        if side is not None:
            if (
                state.stable_side is not None
                and side != state.stable_side
                and state.crossing_point is not None
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
                crossing = (event, state.crossing_point)
            state.stable_side = side
            state.crossing_point = None
        state.last_anchor = anchor
        state.last_seen = observation.timestamp_seconds
        return crossing
