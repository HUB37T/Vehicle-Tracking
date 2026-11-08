"""CSV and atomic JSON outputs, shared by the CLI pipeline and future UI."""

import csv
import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vehicle_counter.contracts import DIRECTIONS, VEHICLE_CLASSES, CountEvent, CountSummary


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Replace metadata atomically so a failed run cannot leave half-written JSON."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def export_counts(
    output_dir: Path,
    *,
    run_id: str,
    line_id: str,
    events: Iterable[CountEvent],
    summary: CountSummary,
) -> dict[str, str]:
    """Validate event/summary consistency, then write both CSVs including zero buckets."""
    events = tuple(events)
    derived = {name: dict.fromkeys(DIRECTIONS, 0) for name in VEHICLE_CLASSES}
    seen = set()
    for event in events:
        if event.run_id != run_id or event.line_id != line_id:
            raise ValueError("Event berasal dari run atau garis yang berbeda.")
        if event.class_name not in derived or event.direction not in DIRECTIONS:
            raise ValueError("Kelas atau arah event tidak valid.")
        key = (event.run_id, event.line_id, event.track_id)
        if key in seen:
            raise ValueError("Event log memiliki crossing duplikat untuk Track ID yang sama.")
        seen.add(key)
        derived[event.class_name][event.direction] += 1
    if summary.total != len(events) or summary.by_class_direction != derived:
        raise ValueError("Ringkasan count tidak sesuai event log.")
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path, counts_path = output_dir / "events.csv", output_dir / "counts.csv"
    with events_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "line_id",
                "track_id",
                "frame_index",
                "timestamp_seconds",
                "class_name",
                "direction",
            ],
        )
        writer.writeheader()
        writer.writerows(asdict(event) for event in events)
    with counts_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["run_id", "line_id", "class_name", "direction", "count"]
        )
        writer.writeheader()
        for name in VEHICLE_CLASSES:
            for direction in DIRECTIONS:
                writer.writerow(
                    {
                        "run_id": run_id,
                        "line_id": line_id,
                        "class_name": name,
                        "direction": direction,
                        "count": derived[name][direction],
                    }
                )
    return {"events_csv": str(events_path), "counts_csv": str(counts_path)}
