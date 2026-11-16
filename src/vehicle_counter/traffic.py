"""Camera-relative traffic state from observed tracks inside a selected road ROI."""

from collections import deque
from dataclasses import asdict, dataclass, replace
from math import hypot
from statistics import median

import numpy as np

from vehicle_counter.contracts import CongestionConfig, TrackObservation
from vehicle_counter.crossing_duplicates import _box_iou

TRAFFIC_LABELS = {
    "insufficient_data": "Belum cukup data",
    "empty": "Sepi",
    "flowing": "Lancar",
    "slow": "Gerak lambat",
    "dense": "Padat",
    "congested": "Indikasi macet",
}


@dataclass(frozen=True)
class TrafficSample:
    frame_index: int
    timestamp_seconds: float
    status: str
    vehicle_count: int
    occupancy_fraction: float
    motion_track_count: int
    slow_fraction: float | None
    median_speed_box_lengths_per_second: float | None
    reason: str


class TrafficMonitor:
    def __init__(
        self, config: CongestionConfig, *, width: int, height: int, max_gap: float, polygon=None
    ):
        import cv2

        self.config = config
        self.polygon = None
        self.zone_mask = np.ones((128, 128), dtype=bool)
        if polygon is not None:
            from vehicle_counter.polygons import validated_polygon

            points = validated_polygon(polygon)
            self.polygon = np.array(
                [(x * (width - 1), y * (height - 1)) for x, y in points], np.float32
            )
            xs, ys = zip(*points, strict=True)
            config = replace(config, roi=(min(xs), min(ys), max(xs), max(ys)))
            self.config = config
        self.roi = tuple(
            v * s for v, s in zip(config.roi, (width, height, width, height), strict=True)
        )
        self.max_gap = max_gap
        if self.polygon is not None:
            self.roi = (
                float(self.polygon[:, 0].min()),
                float(self.polygon[:, 1].min()),
                float(self.polygon[:, 0].max()),
                float(self.polygon[:, 1].max()),
            )
            left, top, right, bottom = self.roi
            scaled = np.rint(
                (self.polygon - (left, top)) / (right - left, bottom - top) * 127
            ).astype(np.int32)
            mask = np.zeros((128, 128), np.uint8)
            cv2.fillPoly(mask, [scaled], 1)
            self.zone_mask = mask.astype(bool)
        self._history: dict[int, deque] = {}
        self.samples: list[TrafficSample] = []
        self.current: TrafficSample | None = None
        self._candidate_since: float | None = None
        self._release_since: float | None = None
        self._congested = False
        self._last_timestamp = -1.0

    def update(self, observations: list[TrackObservation], *, frame_index: int, timestamp: float):
        import cv2

        if timestamp <= self._last_timestamp:
            raise ValueError("Timestamp traffic harus meningkat.")
        self._last_timestamp = timestamp
        left, top, right, bottom = self.roi
        selected = []
        for observation in sorted(observations, key=lambda item: (-item.confidence, item.track_id)):
            x, y = observation.anchor
            if (
                left <= x <= right
                and top <= y <= bottom
                and (self.polygon is None or cv2.pointPolygonTest(self.polygon, (x, y), False) >= 0)
                and all(
                    _box_iou(observation.bbox_xyxy, other.bbox_xyxy) < 0.85 for other in selected
                )
            ):
                selected.append(observation)
        for observation in selected:
            history = self._history.setdefault(observation.track_id, deque())
            if history and timestamp - history[-1][0] > self.max_gap + 1e-9:
                history.clear()
            x1, y1, x2, y2 = observation.bbox_xyxy
            history.append((timestamp, (x1 + x2) / 2, (y1 + y2) / 2, hypot(x2 - x1, y2 - y1)))
            while (
                len(history) > 1
                and timestamp - history[0][0] > self.config.motion_window_seconds + 1e-9
            ):
                history.popleft()
        for track_id in list(self._history):
            if timestamp - self._history[track_id][-1][0] > self.max_gap + 1e-9:
                del self._history[track_id]
        if (
            self.current
            and timestamp - self.current.timestamp_seconds < self.config.sample_seconds - 1e-9
        ):
            return None
        # A bounded raster computes UNION occupancy, preventing double-counted overlap.
        mask = np.zeros((128, 128), dtype=bool)
        for observation in selected:
            x1, y1, x2, y2 = observation.bbox_xyxy
            a = max(0, min(128, int((x1 - left) / (right - left) * 128)))
            b = max(0, min(128, int((y1 - top) / (bottom - top) * 128)))
            c = max(0, min(128, int(np.ceil((x2 - left) / (right - left) * 128))))
            d = max(0, min(128, int(np.ceil((y2 - top) / (bottom - top) * 128))))
            mask[b:d, a:c] = True
        occupancy = float(
            np.count_nonzero(mask & self.zone_mask) / np.count_nonzero(self.zone_mask)
        )
        speeds = []
        for observation in selected:
            history = self._history[observation.track_id]
            duration = history[-1][0] - history[0][0]
            if len(history) >= 3 and duration >= self.config.min_motion_seconds - 1e-9:
                start, end = history[0], history[-1]
                scale = median(item[3] for item in history)
                speeds.append(
                    hypot(end[1] - start[1], end[2] - start[2]) / max(scale, 1) / duration
                )
        slow_fraction = (
            sum(v <= self.config.slow_speed for v in speeds) / len(speeds) if speeds else None
        )
        speed = median(speeds) if speeds else None
        crowded = (
            len(selected) >= self.config.min_vehicles and occupancy >= self.config.min_occupancy
        )
        sufficient = len(speeds) >= max(1, (len(selected) + 1) // 2)
        candidate = crowded and sufficient and slow_fraction >= self.config.min_slow_fraction
        if candidate:
            if self._candidate_since is None:
                self._candidate_since = timestamp
            self._release_since = None
            if timestamp - self._candidate_since >= self.config.hold_seconds - 1e-9:
                self._congested = True
        else:
            self._candidate_since = None
            if self._release_since is None:
                self._release_since = timestamp
            if (
                not sufficient
                or timestamp - self._release_since >= self.config.recovery_seconds - 1e-9
            ):
                self._congested = False
        if not selected:
            status, reason = "empty", "Tidak ada track teramati di area jalan."
        elif not sufficient:
            status, reason = "insufficient_data", "Riwayat gerak belum cukup atau ID terputus."
        elif self._congested:
            status, reason = (
                "congested",
                "Area padat dan sebagian besar kendaraan bergerak lambat selama ambang durasi.",
            )
        elif crowded:
            status, reason = (
                "dense",
                "Area padat. Syarat gerak lambat dan durasi macet belum terpenuhi.",
            )
        elif slow_fraction >= self.config.min_slow_fraction:
            status, reason = (
                "slow",
                "Gerak kendaraan lambat. Ambang kepadatan untuk indikasi macet belum terpenuhi.",
            )
        else:
            status, reason = "flowing", "Syarat kepadatan untuk indikasi macet belum terpenuhi."
        self.current = TrafficSample(
            frame_index,
            timestamp,
            status,
            len(selected),
            occupancy,
            len(speeds),
            slow_fraction,
            speed,
            reason,
        )
        self.samples.append(self.current)
        return self.current

    def episode_rows(self, duration: float | None = None) -> list[dict]:
        end = self.samples[-1].timestamp_seconds if self.samples else 0.0
        duration = end if duration is None else duration
        if duration < end:
            raise ValueError("Durasi video lebih pendek dari sampel lalu lintas.")
        episodes = []
        current = None
        for index, sample in enumerate(self.samples):
            next_time = (
                self.samples[index + 1].timestamp_seconds
                if index + 1 < len(self.samples)
                else duration
            )
            if sample.status == "congested":
                if current is None:
                    current = {
                        "start_seconds": sample.timestamp_seconds,
                        "end_seconds": next_time,
                        "duration_seconds": 0.0,
                        "start_frame_index": sample.frame_index,
                        "peak_vehicle_count": sample.vehicle_count,
                        "ongoing_at_end": False,
                    }
                current["end_seconds"] = next_time
                current["duration_seconds"] = next_time - current["start_seconds"]
                current["peak_vehicle_count"] = max(
                    current["peak_vehicle_count"], sample.vehicle_count
                )
            elif current is not None:
                episodes.append(current)
                current = None
        if current is not None:
            current["ongoing_at_end"] = True
            episodes.append(current)
        return episodes

    def summary(self, duration: float | None = None) -> dict:
        # Durations are measured on the video timeline between consecutive samples.
        durations = dict.fromkeys(TRAFFIC_LABELS, 0.0)
        for first, second in zip(self.samples, self.samples[1:], strict=False):
            durations[first.status] += second.timestamp_seconds - first.timestamp_seconds
        if duration is not None and self.current:
            if duration < self.current.timestamp_seconds:
                raise ValueError("Durasi video lebih pendek dari sampel lalu lintas.")
            durations[self.current.status] += duration - self.current.timestamp_seconds
        return {
            "last_status": self.current.status if self.current else "insufficient_data",
            "durations_seconds": durations,
            "sample_count": len(self.samples),
            "method": "roi_occupancy_and_box_normalized_observed_motion",
            "speed_unit": "box_diagonals_per_second",
            "camera_assumption": "fixed_camera",
            "duplicate_box_iou": 0.85,
            "config": asdict(self.config),
            "episodes": self.episode_rows(duration),
            "duration_basis": "sample_hold_until_next_sample_or_analyzed_end",
        }

    def draw_overlay(self, frame):
        import cv2

        left, top, right, bottom = map(round, self.roi)
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 190, 30), 2)
        if self.current is None:
            return frame
        sample = self.current
        label = TRAFFIC_LABELS[sample.status].upper()
        text = (
            f"{label}  /  {sample.vehicle_count} track  /  "
            f"area {100 * sample.occupancy_fraction:.0f}%"
        )
        height, width = frame.shape[:2]
        scale = max(0.55, min(width / 1400, height / 900))
        text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
        if text_width > width - 16:
            scale *= max(1, width - 16) / text_width
            text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
        y = height - 16
        cv2.rectangle(
            frame,
            (0, y - text_height - 12),
            (min(width, text_width + 20), height),
            (25, 25, 25),
            -1,
        )
        color = (50, 70, 255) if sample.status == "congested" else (255, 220, 80)
        cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        return frame
