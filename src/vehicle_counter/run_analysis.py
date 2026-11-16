"""Explain observed counting conditions and aggregate the accepted event log."""

import csv
from collections import Counter
from dataclasses import asdict
from math import ceil
from pathlib import Path

from vehicle_counter.contracts import DIRECTIONS, VEHICLE_CLASSES, CountEvent, TrackObservation
from vehicle_counter.geometry import PixelCountingLine


class CrossingDiagnostics:
    def __init__(self, line: PixelCountingLine, *, max_gap: float):
        self.line = line
        self.max_gap = max_gap
        self.detection_instances = 0
        self.observation_instances = 0
        self.sides = Counter()
        self.seen_ids: set[int] = set()
        self.last_seen: dict[int, tuple[float, int]] = {}
        self.gap_resets = 0
        self.examples: list[dict] = []

    def update(self, observations: list[TrackObservation], detections) -> None:
        if detections is not None:
            self.detection_instances += len(detections)
        self.observation_instances += len(observations)
        for observation in observations:
            self.seen_ids.add(observation.track_id)
            side = self.line.side(observation.anchor)
            self.sides[side or "neutral"] += 1
            previous = self.last_seen.get(observation.track_id)
            if previous and observation.timestamp_seconds - previous[0] > self.max_gap + 1e-9:
                self.gap_resets += 1
                if len(self.examples) < 6:
                    self.examples.append(
                        {
                            "reason": "observation_gap",
                            "track_id": observation.track_id,
                            "frame_index": observation.frame_index,
                            "previous_frame_index": previous[1],
                            "gap_seconds": observation.timestamp_seconds - previous[0],
                        }
                    )
            self.last_seen[observation.track_id] = (
                observation.timestamp_seconds,
                observation.frame_index,
            )

    def finish(self, event_count: int) -> dict:
        messages = []
        if event_count == 0:
            if self.detection_instances == 0 and self.observation_instances == 0:
                messages.append(
                    "Belum ada deteksi atau track kendaraan. "
                    "Periksa ukuran objek dan masukan detektor."
                )
            elif self.observation_instances == 0:
                messages.append(
                    "Deteksi ada, tetapi belum membentuk track terkonfirmasi untuk counting."
                )
            elif not self.sides["A"] or not self.sides["B"]:
                messages.append(
                    "Track belum teramati di kedua sisi stabil garis. "
                    "Periksa posisi garis pada jalan."
                )
            else:
                messages.append(
                    "Track terlihat di kedua sisi, tetapi belum memenuhi seluruh "
                    "syarat crossing dengan ID yang sama."
                )
            if self.gap_resets:
                messages.append(
                    f"Ada {self.gap_resets} jeda observasi lebih dari "
                    f"{self.max_gap:g} detik yang mereset riwayat gerak."
                )
        return {
            "detection_instances": self.detection_instances,
            "track_observation_instances": self.observation_instances,
            "unique_track_ids": len(self.seen_ids),
            "observations_by_side": {s: self.sides[s] for s in ("A", "B", "neutral")},
            "gap_resets": self.gap_resets,
            "messages": messages,
            "examples": self.examples,
        }


def timeline_rows(
    events: list[CountEvent], *, duration: float, interval: float = 5.0
) -> list[dict]:
    if interval <= 0 or duration < 0:
        raise ValueError("Interval harus positif dan durasi tidak negatif.")
    buckets = Counter(
        (int(e.timestamp_seconds // interval), e.class_name, e.direction) for e in events
    )
    rows = []
    for index in range(ceil(duration / interval)):
        for name in VEHICLE_CLASSES:
            for direction in DIRECTIONS:
                rows.append(
                    {
                        "start_seconds": index * interval,
                        "end_seconds": min(duration, (index + 1) * interval),
                        "class_name": name,
                        "direction": direction,
                        "count": buckets[index, name, direction],
                    }
                )
    if sum(row["count"] for row in rows) != len(events):
        raise ValueError("Ada event di luar durasi timeline.")
    return rows


def write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_analysis(output: Path, events, duration: float, diagnostics, traffic) -> dict[str, str]:
    from vehicle_counter.exporting import write_json

    timeline = timeline_rows(list(events), duration=duration)
    timeline_path = output / "timeline.csv"
    write_rows(
        timeline_path,
        timeline,
        ["start_seconds", "end_seconds", "class_name", "direction", "count"],
    )
    diagnostic_path = output / "diagnostics.json"
    write_json(diagnostic_path, diagnostics)
    artifacts = {"timeline_csv": str(timeline_path), "diagnostics_json": str(diagnostic_path)}
    if traffic is not None:
        from vehicle_counter.traffic import TrafficSample

        path = output / "traffic.csv"
        write_rows(
            path,
            [asdict(sample) for sample in traffic.samples],
            list(TrafficSample.__dataclass_fields__),
        )
        artifacts["traffic_csv"] = str(path)
        path = output / "traffic_episodes.csv"
        write_rows(
            path,
            traffic.episode_rows(duration),
            [
                "start_seconds",
                "end_seconds",
                "duration_seconds",
                "start_frame_index",
                "peak_vehicle_count",
                "ongoing_at_end",
            ],
        )
        artifacts["traffic_episodes_csv"] = str(path)
    return artifacts


def export_zone_analysis(output: Path, zones, duration: float):
    from vehicle_counter.traffic import TrafficSample

    samples, episodes = [], []
    for zone, monitor in zones:
        identity = {"zone_id": zone.zone_id, "zone_name": zone.name}
        samples.extend({**identity, **asdict(sample)} for sample in monitor.samples)
        episodes.extend({**identity, **episode} for episode in monitor.episode_rows(duration))
    paths = {}
    for key, filename, rows, fields in (
        (
            "traffic_zones_csv",
            "traffic_zones.csv",
            samples,
            ["zone_id", "zone_name", *TrafficSample.__dataclass_fields__],
        ),
        (
            "traffic_zone_episodes_csv",
            "traffic_zone_episodes.csv",
            episodes,
            [
                "zone_id",
                "zone_name",
                "start_seconds",
                "end_seconds",
                "duration_seconds",
                "start_frame_index",
                "peak_vehicle_count",
                "ongoing_at_end",
            ],
        ),
    ):
        path = output / filename
        write_rows(path, rows, fields)
        paths[key] = str(path)
    return paths
