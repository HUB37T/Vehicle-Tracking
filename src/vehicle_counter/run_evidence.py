"""A frame-aligned view of the preprocessing, detections and tracks of one run."""

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from vehicle_counter.contracts import Point, PreprocessConfig, TrackObservation
from vehicle_counter.preprocessing import apply_clahe_luminance
from vehicle_counter.tracking import VehicleDetection
from vehicle_counter.visualization import FrameAnnotator


def write_evidence(path: Path, panel: NDArray[np.uint8]) -> None:
    ok, encoded = cv2.imencode(".png", panel)
    if not ok:
        raise RuntimeError("Gagal mengodekan bukti proses.")
    path.write_bytes(encoded.tobytes())


def render_evidence(
    source: NDArray[np.uint8],
    model_input: NDArray[np.uint8],
    detections: Sequence[VehicleDetection] | None,
    observations: Sequence[TrackObservation],
    annotator: FrameAnnotator,
    config: PreprocessConfig,
    *,
    frame_index: int,
    timestamp_seconds: float,
    detection_roi_pixels: tuple[int, int, int, int] | None = None,
    tracker_name: str = "ByteTrack",
) -> NDArray[np.uint8]:
    source_height, source_width = source.shape[:2]
    width, height = 520, 292
    scale = min(width / source_width, height / source_height)
    w, h = max(1, round(source_width * scale)), max(1, round(source_height * scale))
    offset_x, offset_y = (width - w) // 2, (height - h) // 2

    def thumbnail(frame):
        frame_height, frame_width = frame.shape[:2]
        ratio = min(width / frame_width, height / frame_height)
        fw, fh = max(1, round(frame_width * ratio)), max(1, round(frame_height * ratio))
        fx, fy = (width - fw) // 2, (height - fh) // 2
        small = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_AREA)
        if small.ndim == 2:
            small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        panel = np.full((height, width, 3), 18, dtype=np.uint8)
        panel[fy : fy + fh, fx : fx + fw] = small
        return panel

    def point(xy: Point) -> tuple[int, int]:
        return round(xy[0] * scale) + offset_x, round(xy[1] * scale) + offset_y

    def box(panel, coords, label, color):
        x1, y1, x2, y2 = coords
        left, top = point((x1, y1))
        cv2.rectangle(panel, (left, top), point((x2, y2)), color, 2)
        position = (max(2, left), max(15, top - 4))
        cv2.putText(panel, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 3)
        cv2.putText(panel, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)

    clahe_active = config.mode in ("clahe", "bilateral_clahe")
    panels = [("Frame sumber", "Video didekode sebagai BGR 8-bit", thumbnail(source))]
    kernel = config.gaussian_kernel_size
    filter_details = {
        "gamma": ("Gamma correction", f"Gamma {config.gamma:g} | transformasi intensitas"),
        "gaussian": ("Gaussian smoothing", f"Kernel {kernel} x {kernel} | penghalusan"),
        "unsharp": (
            "Unsharp masking",
            f"Kernel {kernel} x {kernel} | amount {config.unsharp_amount:g}",
        ),
        "median": ("Filter median", f"Kernel {kernel} x {kernel} | pengurangan noise"),
    }
    if clahe_active:
        luminance_source = source
        luminance_note = "Kanal L: kecerahan, dipisahkan dari warna"
        clahe_title = "CLAHE | L sesudah"
        if config.mode == "bilateral_clahe":
            luminance_source = cv2.bilateralFilter(
                source,
                config.bilateral_diameter,
                config.bilateral_sigma_color,
                config.bilateral_sigma_space,
            )
            luminance_note = f"Bilateral lebih dahulu | diameter {config.bilateral_diameter}"
            clahe_title = "Bilateral + CLAHE | L sesudah"
        luminance = cv2.cvtColor(luminance_source, cv2.COLOR_BGR2LAB)[:, :, 0]
        grid_x, grid_y = config.clahe_tile_grid_size
        panels.extend(
            (
                ("Ruang warna | BGR ke Lab", luminance_note, thumbnail(luminance)),
                (
                    clahe_title,
                    f"clip limit {config.clahe_clip_limit:g} | grid {grid_x} x {grid_y}",
                    thumbnail(apply_clahe_luminance(luminance, config)),
                ),
            )
        )
    elif config.mode != "none":
        title, note = filter_details[config.mode]
        panels.append((title, note, thumbnail(model_input)))

    if detection_roi_pixels is not None:
        x1, y1, x2, y2 = detection_roi_pixels
        panels.append(
            (
                "Input detektor",
                f"Potongan ROI {x2 - x1} x {y2 - y1} | mode {config.mode}",
                thumbnail(model_input[y1:y2, x1:x2]),
            )
        )
    elif clahe_active:
        panels.append(("Input detektor", "Hasil dikonversi kembali ke BGR", thumbnail(model_input)))

    detection_panel = thumbnail(model_input)
    tracking_panel = thumbnail(source)
    for detection in detections or ():
        box(
            detection_panel,
            detection.bbox_xyxy,
            f"{detection.class_name} {detection.confidence:.2f}",
            (255, 210, 80),
        )
    cv2.line(tracking_panel, point(annotator.line.p1), point(annotator.line.p2), (0, 215, 255), 2)
    for track in observations:
        color = (85, 240, 135)
        trail = [point(p) for p in annotator.trail_for(track.track_id)]
        if len(trail) >= 2:
            cv2.polylines(tracking_panel, [np.array(trail, np.int32)], False, color, 2)
        box(tracking_panel, track.bbox_xyxy, f"{track.class_name} #{track.track_id}", color)
        cv2.circle(tracking_panel, point(track.anchor), 3, (30, 65, 255), -1)

    panels.extend(
        (
            (
                "Deteksi YOLO" if detections is not None else "Detektor uji",
                f"{len(detections)} deteksi | kelas dan confidence"
                if detections is not None
                else "Keluaran deteksi tidak tersedia",
                detection_panel,
            ),
            (
                f"Tracking {tracker_name}" if detections is not None else "Tracker uji",
                f"{len(observations)} track | ID, jejak, dan garis hitung",
                tracking_panel,
            ),
        )
    )
    columns = 2 if len(panels) == 4 else 3
    rows = (len(panels) + columns - 1) // columns
    margin, gap, header, title_height = 18, 12, 66, 58
    canvas = np.full(
        (
            header + rows * (title_height + height + gap) + margin,
            2 * margin + columns * width + (columns - 1) * gap,
            3,
        ),
        (27, 23, 20),
        dtype=np.uint8,
    )

    def text(value, x, y, size, color):
        cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)

    text("BUKTI PROSES  /  PENGHITUNG KENDARAAN", margin, 29, 0.76, (245, 245, 245))
    text(
        f"Frame {frame_index}  |  {timestamp_seconds:.2f} s  |  Prapemrosesan: {config.mode}",
        margin,
        50,
        0.48,
        (205, 192, 173),
    )
    for index, (title, subtitle, panel) in enumerate(panels):
        row, col = divmod(index, columns)
        x = margin + col * (width + gap)
        y = header + row * (title_height + height + gap)
        canvas[y : y + title_height, x : x + width] = (47, 39, 33)
        text(f"{index + 1:02d}  {title}", x + 12, y + 23, 0.62, (245, 240, 230))
        text(subtitle, x + 12, y + 44, 0.47, (220, 200, 180))
        canvas[y + title_height : y + title_height + height, x : x + width] = panel
    return canvas
