"""Original-frame annotation with bounded, per-run track trails."""

from collections import deque
from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from vehicle_counter.contracts import CountSummary, Point, TrackObservation
from vehicle_counter.geometry import PixelCountingLine


class FrameAnnotator:
    def __init__(self, line: PixelCountingLine, *, max_trail_points: int = 30) -> None:
        self.line = line
        self._max_trail_points = max_trail_points
        self._trails: dict[int, deque] = {}
        self._last_seen: dict[int, float] = {}

    def trail_for(self, track_id: int) -> tuple[Point, ...]:
        return tuple(self._trails.get(track_id, ()))

    def render(
        self,
        frame_bgr: NDArray[np.uint8],
        observations: Sequence[TrackObservation],
        summary: CountSummary,
        *,
        timestamp_seconds: float,
    ) -> NDArray[np.uint8]:
        frame = frame_bgr.copy()
        height, width = frame.shape[:2]
        for track_id in list(self._last_seen):
            if timestamp_seconds - self._last_seen[track_id] > 2:
                self._last_seen.pop(track_id)
                self._trails.pop(track_id)
        p1, p2 = tuple(map(round, self.line.p1)), tuple(map(round, self.line.p2))
        cv2.line(frame, p1, p2, (0, 220, 255), 2)
        midpoint = (np.array(p1) + np.array(p2)) / 2
        vector = np.array(p2) - np.array(p1)
        normal = np.array([-vector[1], vector[0]]) / np.linalg.norm(vector)
        for sign, label in ((-1, "A"), (1, "B")):
            position = np.clip(midpoint + sign * 25 * normal, [0, 12], [width - 15, height - 2])
            cv2.putText(
                frame,
                label,
                tuple(map(round, position)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 255),
                2,
            )
        for observation in observations:
            color = (60, 230, 90)
            x1, y1, x2, y2 = map(round, observation.bbox_xyxy)
            anchor = tuple(map(round, observation.anchor))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{observation.class_name} #{observation.track_id} {observation.confidence:.2f}"
            label_y = y2 - 8 if y1 < 110 else y1 - 5
            label_position = (max(0, x1), max(14, min(height - 5, label_y)))
            for text_color, thickness in (((0, 0, 0), 3), (color, 1)):
                cv2.putText(
                    frame,
                    label,
                    label_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    text_color,
                    thickness,
                )
            cv2.circle(frame, anchor, 3, (0, 0, 255), -1)
            trail = self._trails.setdefault(
                observation.track_id, deque(maxlen=self._max_trail_points)
            )
            trail.append(anchor)
            self._last_seen[observation.track_id] = timestamp_seconds
            if len(trail) > 1:
                cv2.polylines(frame, [np.array(trail, dtype=np.int32)], False, color, 1)
        direction_totals = {
            direction: sum(values[direction] for values in summary.by_class_direction.values())
            for direction in ("A_TO_B", "B_TO_A")
        }
        lines = [
            f"TOTAL {summary.total} | A>B {direction_totals['A_TO_B']} | "
            f"B>A {direction_totals['B_TO_A']}",
            *[
                f"{name}: {values['A_TO_B']} / {values['B_TO_A']}"
                for name, values in summary.by_class_direction.items()
            ],
        ]
        panel_width = min(
            width - 1,
            max(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0] for text in lines)
            + 16,
        )
        cv2.rectangle(frame, (0, 0), (panel_width, 20 + len(lines) * 18 - 8), (25, 25, 25), -1)
        for index, text in enumerate(lines):
            y = 20 + index * 18
            cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return frame
