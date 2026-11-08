"""YOLO detection followed by explicit ByteTrack association, once per frame."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

import numpy as np
import yaml
from numpy.typing import NDArray

from vehicle_counter.contracts import BBox, RunConfig, TrackObservation, VideoMetadata
from vehicle_counter.exporting import file_sha256, write_json


@dataclass(frozen=True, slots=True)
class VehicleDetection:
    class_name: str
    confidence: float
    bbox_xyxy: BBox


class FrameTracker(Protocol):
    metadata: dict[str, Any]
    warnings: list[str]

    def track_frame(
        self, frame_bgr: NDArray[np.uint8], *, frame_index: int, timestamp_seconds: float
    ) -> list[TrackObservation]: ...


def resolve_weights(value: str) -> Path:
    path = Path(value).expanduser()
    candidates = [Path("models") / path, path] if path.parent == Path(".") else [path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Bobot {value} belum tersedia. Jalankan: "
        "python -m vehicle_counter.cli prepare-model, atau atur model_weights ke berkas lokal."
    )


def prepare_model(directory: str | Path = "models", *, source: str = "huggingface") -> Path:
    """Explicitly download the official YOLO11n weights, never during video analysis."""
    if source not in ("huggingface", "github"):
        raise ValueError("Sumber bobot harus huggingface atau github.")
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "yolo11n.pt"
    if target.is_file():
        if target.stat().st_size < 1024 * 1024:
            raise ValueError(
                "Bobot lokal terlalu kecil; periksa berkas atau gunakan direktori lain."
            )
        return target
    if source == "github":
        from ultralytics.utils.downloads import attempt_download_asset

        downloaded = Path(attempt_download_asset(str(target), release="v8.3.0"))
        provenance = {
            "source_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
            "release": "v8.3.0",
        }
    elif source == "huggingface":
        import requests

        info_response = requests.get(
            "https://huggingface.co/api/models/Ultralytics/YOLO11?blobs=true", timeout=30
        )
        info_response.raise_for_status()
        info = info_response.json()
        revision = info["sha"]
        entry = next(item for item in info["siblings"] if item["rfilename"] == "yolo11n.pt")
        expected_sha = entry["lfs"]["sha256"]
        url = f"https://huggingface.co/Ultralytics/YOLO11/resolve/{revision}/yolo11n.pt"
        temporary = target.with_suffix(".pt.download")
        with requests.get(url, stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    handle.write(block)
        if file_sha256(temporary) != expected_sha:
            raise RuntimeError("Checksum unduhan bobot berbeda dari metadata resmi Hugging Face.")
        temporary.replace(target)
        downloaded = target
        provenance = {"source_url": url, "revision": revision, "expected_sha256": expected_sha}
    if not downloaded.is_file() or downloaded.stat().st_size < 1024 * 1024:
        raise RuntimeError("Unduhan bobot YOLO11n tidak menghasilkan berkas yang valid.")
    write_json(
        target.with_suffix(".pt.source.json"),
        {
            **provenance,
            "sha256": file_sha256(downloaded),
        },
    )
    return downloaded.resolve()


def load_tracker_settings(value: str) -> tuple[Path, dict[str, Any]]:
    path = Path(value).expanduser()
    if not path.is_file():
        import ultralytics

        if path.name != value:
            raise FileNotFoundError(f"Konfigurasi tracker tidak ditemukan: {value}")
        path = Path(ultralytics.__file__).parent / "cfg" / "trackers" / value
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict) or settings.get("tracker_type") != "bytetrack":
        raise ValueError("Baseline ini memerlukan konfigurasi tracker_type: bytetrack.")
    required = {
        "track_high_thresh",
        "track_low_thresh",
        "new_track_thresh",
        "track_buffer",
        "match_thresh",
        "fuse_score",
    }
    if missing := required - settings.keys():
        raise ValueError(f"Field konfigurasi ByteTrack belum lengkap: {sorted(missing)}")
    for key in ("track_high_thresh", "track_low_thresh", "new_track_thresh", "match_thresh"):
        number = settings[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise ValueError(f"{key} harus berupa angka pada rentang [0, 1].")
        if not np.isfinite(number) or not 0 <= number <= 1:
            raise ValueError(f"{key} harus berada pada rentang [0, 1].")
    if settings["track_low_thresh"] >= settings["track_high_thresh"]:
        raise ValueError("track_low_thresh harus lebih kecil dari track_high_thresh.")
    if type(settings["track_buffer"]) is not int or settings["track_buffer"] < 1:
        raise ValueError("track_buffer harus bilangan bulat positif dalam satuan frame.")
    if not isinstance(settings["fuse_score"], bool):
        raise ValueError("fuse_score harus boolean.")
    return path.resolve(), settings


def observations_from_tracks(
    rows: NDArray,
    detections: Any,
    class_names: dict[int, str],
    *,
    frame_index: int,
    timestamp_seconds: float,
) -> list[TrackObservation]:
    """Use matched detection indices, not Kalman-only positions, for counter anchors."""
    observations = []
    for row in rows:
        if len(row) != 8 or not np.isfinite(row).all():
            raise ValueError(
                "Format keluaran ByteTrack harus [xyxy, id, score, class, detection_idx]."
            )
        track_id, detection_index = int(row[4]), int(row[7])
        if (
            track_id != row[4]
            or detection_index != row[7]
            or not 0 <= detection_index < len(detections)
        ):
            raise ValueError("ByteTrack mengembalikan ID atau indeks deteksi yang tidak valid.")
        class_id = int(detections.cls[detection_index])
        if class_id not in class_names:
            raise ValueError(f"Kelas deteksi {class_id} tidak dikenal model.")
        observations.append(
            TrackObservation(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                track_id=track_id,
                class_name=class_names[class_id],
                confidence=float(detections.conf[detection_index]),
                bbox_xyxy=tuple(map(float, detections.xyxy[detection_index])),
            )
        )
    return observations


class YoloByteTracker:
    """Own a fresh detector and ByteTrack instance for exactly one sequential run."""

    def __init__(self, config: RunConfig, video: VideoMetadata) -> None:
        from ultralytics import YOLO
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils.torch_utils import select_device

        self.detection_roi_pixels = None
        if config.detection_roi is not None:
            left, top, right, bottom = config.detection_roi
            x1, x2 = round(left * video.width), round(right * video.width)
            y1, y2 = round(top * video.height), round(bottom * video.height)
            if min(x2 - x1, y2 - y1) < 2:
                raise ValueError("detection_roi terlalu kecil pada resolusi video ini.")
            for x, y in (config.counting_line.p1, config.counting_line.p2):
                if not (x1 <= x * (video.width - 1) < x2 and y1 <= y * (video.height - 1) < y2):
                    raise ValueError("Seluruh garis hitung harus berada di dalam detection_roi.")
            self.detection_roi_pixels = (x1, y1, x2, y2)
        weights = resolve_weights(config.model_weights)
        tracker_path, settings = load_tracker_settings(config.tracker_config)
        try:
            self._device = str(select_device(config.device, verbose=False))
        except (ValueError, AssertionError, RuntimeError) as error:
            raise ValueError(
                f"Perangkat {config.device} tidak tersedia. Gunakan device: cpu "
                "atau pasang PyTorch yang sesuai GPU."
            ) from error
        self._model = YOLO(str(weights))
        if self._model.task != "detect":
            raise ValueError("Bobot model harus model deteksi, bukan klasifikasi/segmentasi/pose.")
        self._names = dict(self._model.names)
        inverse = {name: index for index, name in self._names.items()}
        if missing := set(config.vehicle_classes) - inverse.keys():
            raise ValueError(f"Kelas kendaraan tidak tersedia pada model: {sorted(missing)}")
        self._class_ids = [inverse[name] for name in config.vehicle_classes]
        self._tracker = BYTETracker(args=SimpleNamespace(**settings))
        self._config = config
        self._video = video
        self._next_frame = 0
        self.last_detections: tuple[VehicleDetection, ...] = ()
        self.warnings = []
        if config.detector_conf > settings["track_low_thresh"]:
            self.warnings.append(
                "detector_conf melebihi track_low_thresh; sebagian kandidat confidence rendah "
                "tidak mencapai asosiasi ByteTrack."
            )
        self.metadata = {
            "kind": "yolo-bytetrack",
            "is_model_inference": True,
            "weights_path": str(weights),
            "weights_sha256": file_sha256(weights),
            "class_names": self._names,
            "effective_device": self._device,
            "tracker_path": str(tracker_path),
            "tracker_settings": settings,
            "track_buffer_units": "frames",
            "anchor_source": "matched_detection_bbox",
        }
        if self.detection_roi_pixels is not None:
            self.metadata["detection_roi_pixels"] = self.detection_roi_pixels
        provenance = weights.with_suffix(".pt.source.json")
        if provenance.is_file():
            self.metadata["weights_provenance"] = json.loads(provenance.read_text(encoding="utf-8"))

    def track_frame(
        self, frame_bgr: NDArray[np.uint8], *, frame_index: int, timestamp_seconds: float
    ) -> list[TrackObservation]:
        if frame_index != self._next_frame:
            raise ValueError("Tracker memerlukan setiap frame berurutan, dimulai dari 0.")
        if frame_bgr.dtype != np.uint8 or frame_bgr.shape != (
            self._video.height,
            self._video.width,
            3,
        ):
            raise ValueError("Frame tracker harus BGR uint8 dengan dimensi video asli.")
        if not np.isclose(timestamp_seconds, frame_index / self._video.source_fps):
            raise ValueError("Timestamp harus mengikuti frame_index / FPS sumber CFR.")
        detector_frame = frame_bgr
        if self.detection_roi_pixels is not None:
            x1, y1, x2, y2 = self.detection_roi_pixels
            detector_frame = frame_bgr[y1:y2, x1:x2]
        results = self._model.predict(
            source=detector_frame,
            classes=self._class_ids,
            conf=self._config.detector_conf,
            imgsz=self._config.imgsz,
            device=self._device,
            verbose=False,
            save=False,
        )
        detections = results[0].boxes.cpu().numpy()
        if self.detection_roi_pixels is not None:
            from ultralytics.engine.results import Boxes

            data = detections.data.copy()
            data[:, [0, 2]] += x1
            data[:, [1, 3]] += y1
            detections = Boxes(data, frame_bgr.shape[:2])
        self.last_detections = tuple(
            VehicleDetection(self._names[int(class_id)], float(score), tuple(map(float, box)))
            for box, score, class_id in zip(
                detections.xyxy, detections.conf, detections.cls, strict=True
            )
        )
        # Update even when there are zero detections, so lost-track timers advance.
        rows = self._tracker.update(detections, img=frame_bgr)
        observations = observations_from_tracks(
            rows,
            detections,
            self._names,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
        )
        self._next_frame += 1
        return observations
