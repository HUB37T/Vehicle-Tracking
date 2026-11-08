"""Export a visual comparison of classical PCD stages on consecutive video frames."""

from dataclasses import asdict, replace
from pathlib import Path
from tempfile import mkdtemp

import cv2
import numpy as np
from numpy.typing import NDArray

from vehicle_counter.contracts import PREPROCESS_MODES, PreprocessConfig
from vehicle_counter.exporting import write_json
from vehicle_counter.preprocessing import preprocess_frame
from vehicle_counter.video import VideoReader


def _optical_flow(previous, current, frame):
    overlay = frame.copy()
    corners = cv2.goodFeaturesToTrack(previous, maxCorners=300, qualityLevel=0.01, minDistance=7)
    if corners is None:
        return overlay, 0
    predicted, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, corners, None)
    if predicted is None or status is None:
        return overlay, 0
    height, width = current.shape
    arrows = 0
    valid = status.ravel() == 1
    for start, end in zip(
        corners.reshape(-1, 2)[valid], predicted.reshape(-1, 2)[valid], strict=True
    ):
        if not (0 <= end[0] < width and 0 <= end[1] < height):
            continue
        displacement = end - start
        if np.linalg.norm(displacement) < 1:
            continue
        origin = tuple(np.rint(start).astype(int))
        tip = tuple(np.rint(start + 3 * displacement).astype(int))
        cv2.arrowedLine(overlay, origin, tip, (0, 255, 255), 2, tipLength=0.25)
        arrows += 1
    return overlay, arrows


def _motion_stages(previous_frame, frame, min_area):
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    previous_blur = cv2.GaussianBlur(previous_gray, (5, 5), 0)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    difference = cv2.absdiff(previous_blur, blurred)
    fixed_threshold = 20
    _, fixed = cv2.threshold(difference, fixed_threshold, 255, cv2.THRESH_BINARY)
    otsu_threshold, otsu = cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    opened = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    components = []
    marked = frame.copy()
    for label in range(1, count):
        x, y, width, height, area = map(int, stats[label])
        if area < min_area:
            continue
        centroid = centroids[label].tolist()
        components.append(
            {
                "label": label,
                "bbox_xywh": [x, y, width, height],
                "area_px": area,
                "centroid_xy": centroid,
            }
        )
        cv2.rectangle(marked, (x, y), (x + width - 1, y + height - 1), (0, 255, 0), 2)
        cv2.circle(marked, tuple(map(round, centroid)), 3, (0, 255, 255), -1)
    flow, arrows = _optical_flow(previous_gray, gray, frame)
    panels = [
        ("Frame sebelumnya", previous_frame),
        ("Frame target", frame),
        ("Grayscale", gray),
        ("Gaussian 5 x 5", blurred),
        ("Selisih antarframe", difference),
        (f"Threshold tetap: {fixed_threshold}", fixed),
        (f"Otsu: {otsu_threshold:g}", otsu),
        ("Opening 3 x 3", opened),
        ("Closing 7 x 7", closed),
        (f"CCL: {len(components)} komponen", marked),
        ("Canny 50 / 150", cv2.Canny(blurred, 50, 150)),
        ("Lucas-Kanade (panah 3x)", flow),
    ]
    report = {
        "fixed_threshold": fixed_threshold,
        "otsu_threshold": otsu_threshold,
        "min_area_px": min_area,
        "motion_components": components,
        "flow_arrows": arrows,
    }
    return panels, report


def _contact_sheet(panels: list[tuple[str, NDArray[np.uint8]]]) -> NDArray[np.uint8]:
    height, width = panels[0][1].shape[:2]
    scale = min(400 / width, 260 / height, 1)
    width, height = max(1, round(width * scale)), max(1, round(height * scale))
    cell_width = max(260, width)
    columns = 3
    cell_height = height + 28
    rows = (len(panels) + columns - 1) // columns
    sheet = np.full((rows * cell_height, columns * cell_width, 3), 24, dtype=np.uint8)
    for index, (title, panel) in enumerate(panels):
        row, column = divmod(index, columns)
        x, y = column * cell_width, row * cell_height
        if panel.ndim == 2:
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        sheet[y + 28 : y + cell_height, x : x + width] = cv2.resize(
            panel,
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
        cv2.putText(
            sheet,
            title,
            (x + 7, y + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return sheet


def export_lab(
    source: Path,
    output_root: Path,
    config: PreprocessConfig,
    *,
    frame_index: int,
    min_area: int,
) -> dict:
    """Save coursework panels and metadata for target frame N and its predecessor N-1."""
    if frame_index < 1 or min_area < 1:
        raise ValueError("Indeks frame target dan luas minimum harus >= 1.")
    with VideoReader(source) as reader:
        previous = None
        for index, frame in reader:
            if index == frame_index:
                break
            previous = frame
        else:
            raise ValueError(f"Frame {frame_index} tidak ditemukan di video.")
        metadata = reader.metadata
    panels, report = _motion_stages(previous, frame, min_area)
    preprocessing = [
        (mode, preprocess_frame(frame, replace(config, mode=mode))) for mode in PREPROCESS_MODES
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix="pcd-lab-", dir=output_root.resolve()))
    for name, images in (("classical.png", panels), ("preprocessing.png", preprocessing)):
        # imencode plus Path handles non-ASCII paths on Windows.
        ok, encoded = cv2.imencode(".png", _contact_sheet(images))
        if not ok:
            raise RuntimeError(f"Gagal membuat gambar {name}.")
        (output / name).write_bytes(encoded.tobytes())
    report.update(
        {
            "video": asdict(metadata),
            "reference_frame_index": frame_index - 1,
            "frame_index": frame_index,
            "timestamp_seconds": frame_index / metadata.source_fps,
            "preprocess_config": asdict(config),
            "preprocess_modes_compared": list(PREPROCESS_MODES),
            "output_dir": str(output),
            "note": "Komponen adalah perubahan piksel. Panah adalah perpindahan titik citra.",
        }
    )
    write_json(output / "lab.json", report)
    return {
        "frame_index": frame_index,
        "motion_component_count": len(report["motion_components"]),
        "flow_arrows": report["flow_arrows"],
        "output_dir": str(output),
    }
