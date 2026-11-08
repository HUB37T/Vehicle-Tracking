"""Appearance evidence for repeated crossing candidates after a Track ID change."""

from dataclasses import dataclass
from math import ceil, floor

import cv2
import numpy as np
from numpy.typing import NDArray

from vehicle_counter.contracts import BBox, CountEvent, TrackObservation
from vehicle_counter.geometry import PixelCountingLine


@dataclass(frozen=True, slots=True)
class _Crossing:
    event: CountEvent
    box: BBox
    patch: NDArray[np.uint8]
    clearance: float


def _box_iou(a: BBox, b: BBox) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    return intersection / (
        (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    )


def _upper_patch(frame: NDArray[np.uint8], box: BBox) -> NDArray[np.uint8] | None:
    x1, y1, x2, y2 = box
    height, width = frame.shape[:2]
    crop = frame[
        max(0, floor(y1)) : max(0, min(height, floor(y1 + (y2 - y1) * 0.65))),
        max(0, floor(x1)) : max(0, min(width, ceil(x2))),
    ]
    if min(crop.shape[:2]) < 12:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    patch = cv2.resize(gray, (48, 32), interpolation=cv2.INTER_AREA)
    # Uniform patches cannot distinguish vehicle appearance.
    return patch if patch[4:22, 7:41].std() >= 8 else None


def _appearance_matches(a: NDArray[np.uint8], b: NDArray[np.uint8]) -> bool:
    # Search in both directions so a small common feature is insufficient by itself.
    return all(
        cv2.matchTemplate(image, template[4:22, 7:41], cv2.TM_CCOEFF_NORMED).max() >= 0.8
        for image, template in ((a, b), (b, a))
    )


class CrossingDuplicates:
    """Compare candidates near the same crossing while a previous vehicle may still be there."""

    def __init__(self, line: PixelCountingLine) -> None:
        self._line = line
        self._recent: dict[int, _Crossing] = {}

    def observe_counted(self, observation: TrackObservation) -> None:
        crossing = self._recent.get(observation.track_id)
        if crossing and abs(self._line.signed_distance(observation.anchor)) > crossing.clearance:
            del self._recent[observation.track_id]

    def suppress(
        self, event: CountEvent, observation: TrackObservation, frame: NDArray[np.uint8]
    ) -> bool:
        for track_id, crossing in list(self._recent.items()):
            if event.timestamp_seconds - crossing.event.timestamp_seconds > 15:
                del self._recent[track_id]
        box = observation.bbox_xyxy
        patch = _upper_patch(frame, box)
        if patch is None:
            return False
        for crossing in self._recent.values():
            if (
                event.direction == crossing.event.direction
                and _box_iou(box, crossing.box) >= 0.65
                and _appearance_matches(patch, crossing.patch)
            ):
                return True
        x1, y1, x2, y2 = box
        corners = ((x1, y1), (x2, y1), (x1, y2), (x2, y2))
        distances = [self._line.signed_distance(point) for point in corners]
        self._recent[event.track_id] = _Crossing(event, box, patch, max(distances) - min(distances))
        return False
