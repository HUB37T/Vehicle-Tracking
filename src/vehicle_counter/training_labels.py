"""Immutable manual labels; export YOLO boxes without inventing train/validation metrics."""

import io
import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from vehicle_counter.contracts import VEHICLE_CLASSES
from vehicle_counter.exporting import write_json


def validate_boxes(boxes):
    if not isinstance(boxes, list) or len(boxes) > 200:
        raise ValueError("Label harus berupa daftar maksimal 200 kotak.")
    result = []
    for box in boxes:
        if (
            not isinstance(box, dict)
            or set(box) != {"class_name", "bbox"}
            or box["class_name"] not in VEHICLE_CLASSES
        ):
            raise ValueError("Kelas/kotak label tidak valid.")
        coords = tuple(box["bbox"])
        if len(coords) != 4 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)
            for v in coords
        ):
            raise ValueError("Kotak memerlukan empat angka berhingga.")
        x1, y1, x2, y2 = coords
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("Kotak harus terurut, luas positif, dan berada pada rentang 0–1.")
        result.append({"class_name": box["class_name"], "bbox": list(coords)})
    return result


def yolo_text(boxes):
    lines = []
    for box in validate_boxes(boxes):
        x1, y1, x2, y2 = box["bbox"]
        lines.append(
            f"{VEHICLE_CLASSES.index(box['class_name'])} "
            f"{(x1 + x2) / 2:.8f} {(y1 + y2) / 2:.8f} {x2 - x1:.8f} {y2 - y1:.8f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def save_labels(
    root: Path,
    *,
    source_sha256: str,
    frame_index: int,
    frame_bgr,
    boxes,
    confirmed: bool,
    source_label: str = "video.mp4",
):
    if confirmed is not True:
        raise ValueError("Periksa seluruh kotak dan konfirmasi dahulu.")
    if not re.fullmatch("[0-9a-f]{64}", source_sha256):
        raise ValueError("Checksum sumber tidak valid.")
    if type(frame_index) is not int or frame_index < 0:
        raise ValueError("Indeks frame tidak valid.")
    boxes = validate_boxes(boxes)
    if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("Frame sumber harus BGR uint8.")
    directory = Path(root) / uuid4().hex
    directory.mkdir(parents=True)
    ok, encoded = cv2.imencode(".png", frame_bgr)
    if not ok:
        raise ValueError("Frame sumber tidak dapat disimpan.")
    (directory / "image.png").write_bytes(encoded.tobytes())
    (directory / "label.txt").write_text(yolo_text(boxes), encoding="utf-8")
    record = {
        "source_sha256": source_sha256,
        "source_label": source_label,
        "frame_index": frame_index,
        "boxes": boxes,
        "width": frame_bgr.shape[1],
        "height": frame_bgr.shape[0],
        "annotation_method": "manual_reviewed",
        "negative_frame": not boxes,
        "classes": list(VEHICLE_CLASSES),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(directory / "annotation.json", record)
    return record


def latest_labels(root):
    selected = {}
    for path in Path(root).glob("*/annotation.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            key = (record["source_sha256"], record["frame_index"])
            if key not in selected or record["created_at_utc"] > selected[key][1]["created_at_utc"]:
                selected[key] = (path.parent, record)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return list(selected.values())


def export_dataset(root):
    records = latest_labels(root)
    if not records:
        raise ValueError("Belum ada label manual.")
    if (
        len(records) > 200
        or sum((p / "image.png").stat().st_size for p, _ in records) > 150 * 1024 * 1024
    ):
        raise ValueError(
            "Ekspor UI dibatasi 200 frame / 150 MB. Kelola kumpulan yang lebih"
            " besar dari folder dataset lokal."
        )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, record in records:
            stem = f"{record['source_sha256']}_{record['frame_index']:08d}"
            archive.write(path / "image.png", f"images/{stem}.png")
            archive.write(path / "label.txt", f"labels/{stem}.txt")
        archive.writestr(
            "metadata.json",
            json.dumps(
                {
                    "classes": list(VEHICLE_CLASSES),
                    "frames": [r for _, r in records],
                    "split_status": "not_assigned",
                },
                indent=2,
            ),
        )
        archive.writestr(
            "README.txt",
            "Label manual format YOLO: class cx cy width height (0-1).\n"
            "Kelas 0=car, 1=motorcycle, 2=bus, 3=truck. Gambar adalah frame sumber tanpa overlay.\n"
            "Pisahkan train/validation/test menurut VIDEO SUMBER, bukan frame berdekatan.\n"
            "Split belum ditetapkan. Jangan menghitung mAP atau menganggap "
            "data uji tetap independen setelah dipakai training.\n",
        )
    return buffer.getvalue()
