"""Short observation history for checking events with looser track association."""

from collections import deque
from statistics import median

from vehicle_counter.contracts import BBox, CountEvent, TrackObservation
from vehicle_counter.crossing_duplicates import _box_iou

_HISTORY_SECONDS = 0.50
_DUPLICATE_SECONDS = 0.35
_CLASS_SECONDS = 0.30


def _boxes_follow_together(a: BBox, b: BBox) -> bool:
    iou = _box_iou(a, b)
    if iou < 0.50:
        return False
    width_a, width_b = a[2] - a[0], b[2] - b[0]
    area_a = width_a * (a[3] - a[1])
    area_b = width_b * (b[3] - b[1])
    intersection = iou * (area_a + area_b) / (1 + iou)
    center_delta = abs((a[0] + a[2]) - (b[0] + b[2])) / 2
    return intersection / min(area_a, area_b) >= 0.85 and center_delta <= 0.28 * min(
        width_a, width_b
    )


class DenseCrossingGuard:
    def __init__(self) -> None:
        self._frames: deque[tuple[float, dict[int, TrackObservation]]] = deque()
        self._accepted: deque[CountEvent] = deque()

    def observe(self, observations: list[TrackObservation], timestamp: float) -> None:
        self._frames.append((timestamp, {item.track_id: item for item in observations}))
        while timestamp - self._frames[0][0] > _HISTORY_SECONDS + 1e-9:
            self._frames.popleft()
        while self._accepted and timestamp - self._accepted[0].timestamp_seconds > (
            _DUPLICATE_SECONDS + 1e-9
        ):
            self._accepted.popleft()

    def recent_class(self, event: CountEvent, observation: TrackObservation) -> str:
        current = observation.class_name
        # A short burst of bus detections can also come from a box truck.
        if current in (event.class_name, "bus"):
            return event.class_name
        recent = [
            frame[event.track_id]
            for timestamp, frame in self._frames
            if event.timestamp_seconds - timestamp <= _CLASS_SECONDS + 1e-9
            and event.track_id in frame
        ]
        scores = [item.confidence for item in recent if item.class_name == current]
        if len(scores) >= 4 and 3 * len(scores) >= 2 * len(recent):
            # A passenger car should not become a truck on weak recent votes alone.
            if (event.class_name, current) != ("car", "truck") or median(scores) >= 0.50:
                return current
        return event.class_name

    def duplicate_of(self, event: CountEvent) -> int | None:
        for other in self._accepted:
            if event.direction != other.direction:
                continue
            same_class = event.class_name == other.class_name
            if not same_class and "motorcycle" in (event.class_name, other.class_name):
                continue
            shared = [
                (timestamp, frame[event.track_id], frame[other.track_id])
                for timestamp, frame in self._frames
                if event.track_id in frame
                and other.track_id in frame
                and (same_class or event.timestamp_seconds - timestamp <= 0.26)
            ]
            if len(shared) < 3 or event.timestamp_seconds - shared[-1][0] > 0.12 + 1e-9:
                continue
            overlaps = [_boxes_follow_together(a.bbox_xyxy, b.bbox_xyxy) for _, a, b in shared]
            hits = sum(overlaps)
            minimum_hits = 3 if same_class else 5
            if overlaps[-1] and hits >= minimum_hits and 3 * hits >= 2 * len(shared):
                return other.track_id
        return None

    def accept(self, event: CountEvent) -> None:
        self._accepted.append(event)
