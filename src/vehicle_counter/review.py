"""Persist manual review notes separately from immutable automatic events."""

import csv
import json
from pathlib import Path
from uuid import uuid4

from vehicle_counter.contracts import DIRECTIONS, VEHICLE_CLASSES
from vehicle_counter.exporting import write_json

REVIEW_STATUSES = ("sesuai", "terhitung_ganda", "salah_kelas", "terlewat", "meragukan")
REVIEW_FIELDS = [
    "review_id",
    "run_id",
    "frame_index",
    "track_id",
    "status",
    "corrected_class",
    "direction",
    "notes",
]


def load_reviews(directory: Path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((directory / "reviews").glob("*.json"))
    ]


def save_review(
    directory: Path,
    *,
    frame_index: int,
    track_id: int | None,
    status: str,
    corrected_class: str,
    direction: str,
    notes: str,
) -> dict:
    manifest = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise ValueError("Tinjauan hanya untuk run yang selesai.")
    if type(frame_index) is not int or not 0 <= frame_index < manifest["processed_frames"]:
        raise ValueError("Frame tinjauan berada di luar hasil analisis.")
    if (
        status not in REVIEW_STATUSES
        or corrected_class not in VEHICLE_CLASSES
        or direction not in DIRECTIONS
    ):
        raise ValueError("Status, kelas, atau arah tinjauan tidak valid.")
    if track_id is not None:
        with (directory / "events.csv").open(encoding="utf-8", newline="") as handle:
            ids = {int(row["track_id"]) for row in csv.DictReader(handle)}
        if type(track_id) is not int or track_id not in ids:
            raise ValueError("Track ID tinjauan tidak ada pada event run ini.")
    if status != "terlewat" and track_id is None:
        raise ValueError("Pilih Track ID event yang akan ditinjau.")
    item = {
        "review_id": uuid4().hex,
        "run_id": manifest["run_id"],
        "frame_index": frame_index,
        "track_id": track_id,
        "status": status,
        "corrected_class": corrected_class,
        "direction": direction,
        "notes": str(notes).strip()[:2000],
    }
    target = directory / "reviews"
    target.mkdir(exist_ok=True)
    # A new immutable record per save keeps concurrent reviewers' notes intact.
    write_json(target / f"{item['review_id']}.json", item)
    return item


def reviews_csv(directory: Path) -> bytes:
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=REVIEW_FIELDS)
    writer.writeheader()
    writer.writerows(load_reviews(directory))
    return output.getvalue().encode("utf-8-sig")
