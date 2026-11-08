"""Evidence for reviewing crossings; flags never change domain counts."""

import csv
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from vehicle_counter.contracts import CountEvent, TrackObservation
from vehicle_counter.tracking import VehicleDetection

AUDIT_FIELDS = (
    "run_id",
    "line_id",
    "track_id",
    "frame_index",
    "timestamp_seconds",
    "class_name",
    "direction",
    "x1",
    "y1",
    "x2",
    "y2",
    "anchor_x",
    "anchor_y",
    "overlapping_boxes",
    "review_reason",
)


def audit_crossings(
    events: Sequence[CountEvent],
    observations: Sequence[TrackObservation],
    detections: Sequence[VehicleDetection] | None,
) -> list[dict]:
    """Flag anchors inside another detection, a candidate for visual occlusion.

    A box overlap cannot prove an error or clear a crossing. Missing detections
    also prevent this check; the reason field makes that absence explicit.
    """
    by_id = {item.track_id: item for item in observations}
    rows = []
    for event in events:
        observation = by_id[event.track_id]
        x, y = observation.anchor
        overlapping = sum(
            detection.bbox_xyxy != observation.bbox_xyxy
            and detection.bbox_xyxy[0] < x < detection.bbox_xyxy[2]
            and detection.bbox_xyxy[1] < y < detection.bbox_xyxy[3]
            for detection in detections or ()
        )
        rows.append(
            {
                **asdict(event),
                **dict(zip(("x1", "y1", "x2", "y2"), observation.bbox_xyxy, strict=True)),
                "anchor_x": x,
                "anchor_y": y,
                "overlapping_boxes": overlapping,
                "review_reason": (
                    "anchor_inside_other_detection"
                    if overlapping
                    else "detections_unavailable"
                    if detections is None
                    else ""
                ),
            }
        )
    return rows


def write_crossing_audit(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
