"""Compare completed crossing runs with manual annotations on the same clip."""

import csv
import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

from vehicle_counter.contracts import DIRECTIONS, VEHICLE_CLASSES


@dataclass(frozen=True)
class Event:
    event_id: str
    frame_index: int
    timestamp_seconds: float
    class_name: str
    direction: str


def read_events(path: Path, *, id_column: str, clip_id: str | None = None) -> list[Event]:
    required = {id_column, "frame_index", "timestamp_seconds", "class_name", "direction"}
    if clip_id is not None:
        required.add("clip_id")
    events = []
    ids = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Kolom wajib CSV tidak lengkap: {path.name}.")
        for row in reader:
            if clip_id is not None and row["clip_id"] != clip_id:
                continue
            try:
                event = Event(
                    row[id_column],
                    int(row["frame_index"]),
                    float(row["timestamp_seconds"]),
                    row["class_name"],
                    row["direction"],
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Data frame atau waktu tidak valid pada baris {reader.line_num}."
                ) from error
            if (
                not event.event_id
                or event.event_id in ids
                or event.frame_index < 0
                or not isfinite(event.timestamp_seconds)
                or event.timestamp_seconds < 0
                or event.class_name not in VEHICLE_CLASSES
                or event.direction not in DIRECTIONS
            ):
                raise ValueError(
                    f"Event tidak valid atau ID duplikat pada baris {reader.line_num}."
                )
            ids.add(event.event_id)
            events.append(event)
    return sorted(events, key=lambda event: (event.timestamp_seconds, event.event_id))


def _match(manual: list[Event], predicted: list[Event], tolerance: float) -> list[tuple[int, int]]:
    """Maximize matched pairs, then minimize total time error, within one class and direction."""
    n, m = len(manual), len(predicted)
    decisions = [bytearray(m) for _ in range(n)]
    next_row = [(0, 0.0)] * (m + 1)
    for i in range(n - 1, -1, -1):
        row = [(0, 0.0)] * (m + 1)
        for j in range(m - 1, -1, -1):
            best, decision = next_row[j], 0
            if row[j + 1] > best:
                best, decision = row[j + 1], 1
            delta = abs(manual[i].timestamp_seconds - predicted[j].timestamp_seconds)
            if delta <= tolerance + 1e-9:
                count, negative_error = next_row[j + 1]
                pair = (count + 1, negative_error - delta)
                if pair >= best:
                    best, decision = pair, 2
            row[j], decisions[i][j] = best, decision
        next_row = row
    pairs = []
    i = j = 0
    while i < n and j < m:
        decision = decisions[i][j]
        if decision == 2:
            pairs.append((i, j))
        i += decision != 1
        j += decision != 0
    return pairs


def evaluate_events(manual: list[Event], predicted: list[Event], tolerance: float = 0.5) -> dict:
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("Toleransi waktu harus berhingga dan minimal nol.")
    buckets, matched, misses, extras = [], [], [], []
    for name in VEHICLE_CLASSES:
        for direction in DIRECTIONS:
            expected, observed = (
                sorted(
                    (e for e in events if (e.class_name, e.direction) == (name, direction)),
                    key=lambda event: event.timestamp_seconds,
                )
                for events in (manual, predicted)
            )
            pairs = _match(expected, observed, tolerance)
            expected_used, observed_used = {i for i, _ in pairs}, {j for _, j in pairs}
            matched.extend(
                {
                    "manual": asdict(expected[i]),
                    "predicted": asdict(observed[j]),
                    "delta_seconds": round(
                        observed[j].timestamp_seconds - expected[i].timestamp_seconds, 6
                    ),
                }
                for i, j in pairs
            )
            misses.extend(asdict(e) for i, e in enumerate(expected) if i not in expected_used)
            extras.extend(asdict(e) for j, e in enumerate(observed) if j not in observed_used)
            buckets.append(
                {
                    "class_name": name,
                    "direction": direction,
                    "manual": len(expected),
                    "predicted": len(observed),
                    "absolute_error": abs(len(expected) - len(observed)),
                    "matched": len(pairs),
                }
            )
    tp, fp, fn = len(matched), len(extras), len(misses)
    errors = sum(bucket["absolute_error"] for bucket in buckets)
    return {
        "matching_basis": "class_direction_time",
        "identity_verified": False,
        "review_note": (
            "Pasangan otomatis hanya memakai kelas, arah, dan waktu. "
            "Periksa identitas pada video, terutama saat kendaraan rapat atau tertutup."
        ),
        "manual_total": len(manual),
        "predicted_total": len(predicted),
        "absolute_total_error": abs(len(predicted) - len(manual)),
        "mae_bucket": errors / len(buckets),
        "count_error_percent": 100 * errors / len(manual) if manual else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "buckets": buckets,
        "matches": matched,
        "missed_events": misses,
        "extra_events": extras,
    }


def compare_runs(gt: Path, clip_id: str, runs: list[Path], tolerance: float = 0.5) -> dict:
    """Bind the CSV to the video checksum and counting line recorded in its sidecar."""
    sidecar = json.loads(gt.with_suffix(".json").read_text(encoding="utf-8"))
    if not isinstance(sidecar, dict) or not {"clip_id", "source_sha256", "counting_line"}.issubset(
        sidecar
    ):
        raise ValueError("Metadata anotasi harus memuat clip_id, source_sha256, dan counting_line.")
    if sidecar["clip_id"] != clip_id:
        raise ValueError("Clip ID berbeda dari metadata anotasi.")
    manual = read_events(gt, id_column="manual_event_id", clip_id=clip_id)
    results = []
    for directory in runs:
        manifest = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if manifest["status"] != "completed":
            raise ValueError(f"Run belum selesai: {directory.name}.")
        if (
            manifest["source"]["sha256"] != sidecar["source_sha256"]
            or manifest["config"]["counting_line"] != sidecar["counting_line"]
        ):
            raise ValueError(f"Video atau garis tidak sesuai dengan anotasi: {directory.name}.")
        predicted = read_events(directory / "events.csv", id_column="track_id")
        if len(predicted) != manifest["summary"]["total"]:
            raise ValueError(f"Jumlah event berbeda dari ringkasan: {directory.name}.")
        results.append(
            {
                "run_id": manifest["run_id"],
                "mode": manifest["config"]["preprocess"]["mode"],
                "processing_fps": manifest["processing_fps"],
                **evaluate_events(manual, predicted, tolerance),
            }
        )
    return {"clip_id": clip_id, "tolerance_seconds": tolerance, "runs": results}
