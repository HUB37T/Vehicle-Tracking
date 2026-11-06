"""Shared data contracts; image coordinates always refer to the original frame."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Real

import numpy as np
from numpy.typing import NDArray

VEHICLE_CLASSES = ("car", "motorcycle", "bus", "truck")
DIRECTIONS = ("A_TO_B", "B_TO_A")
Point = tuple[float, float]
BBox = tuple[float, float, float, float]


def _finite(value: float, name: str, minimum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} harus berupa angka berhingga.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} harus >= {minimum}.")


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} harus bilangan bulat >= {minimum}.")


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} harus berupa teks yang tidak kosong.")


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    mode: str = "none"
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)

    def __post_init__(self) -> None:
        if self.mode not in ("none", "clahe"):
            raise ValueError("Mode prapemrosesan harus none atau clahe.")
        _finite(self.clahe_clip_limit, "clahe_clip_limit", 0)
        if self.clahe_clip_limit == 0:
            raise ValueError("clahe_clip_limit harus lebih besar dari nol.")
        tiles = tuple(self.clahe_tile_grid_size)
        if len(tiles) != 2:
            raise ValueError("clahe_tile_grid_size memerlukan dua bilangan bulat.")
        for value in tiles:
            _integer(value, "clahe_tile_grid_size", 1)
        object.__setattr__(self, "clahe_tile_grid_size", tiles)


@dataclass(frozen=True, slots=True)
class CountingLineConfig:
    line_id: str = "main"
    p1: Point = (0.10, 0.60)
    p2: Point = (0.90, 0.60)
    epsilon_height_fraction: float = 0.01

    def __post_init__(self) -> None:
        _nonempty(self.line_id, "line_id")
        for name in ("p1", "p2"):
            point = tuple(getattr(self, name))
            if len(point) != 2:
                raise ValueError(f"{name} memerlukan dua koordinat.")
            for value in point:
                _finite(value, name, 0)
                if value > 1:
                    raise ValueError(f"{name} harus berada pada rentang [0, 1].")
            object.__setattr__(self, name, point)
        if self.p1 == self.p2:
            raise ValueError("P1 dan P2 harus berbeda.")
        _finite(self.epsilon_height_fraction, "epsilon_height_fraction", 0)
        if self.epsilon_height_fraction > 1:
            raise ValueError("epsilon_height_fraction harus <= 1.")


@dataclass(frozen=True, slots=True)
class RunConfig:
    model_weights: str = "yolo11n.pt"
    tracker_config: str = "bytetrack.yaml"
    device: str = "cpu"
    imgsz: int = 640
    detector_conf: float = 0.10
    vehicle_classes: tuple[str, ...] = VEHICLE_CLASSES
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    counting_line: CountingLineConfig = field(default_factory=CountingLineConfig)
    min_track_hits: int = 3
    max_crossing_gap_seconds: float = 0.5

    def __post_init__(self) -> None:
        for name in ("model_weights", "tracker_config", "device"):
            _nonempty(getattr(self, name), name)
        _integer(self.imgsz, "imgsz", 1)
        _integer(self.min_track_hits, "min_track_hits", 1)
        _finite(self.detector_conf, "detector_conf", 0)
        if self.detector_conf > 1:
            raise ValueError("detector_conf harus <= 1.")
        _finite(self.max_crossing_gap_seconds, "max_crossing_gap_seconds", 0)
        if self.max_crossing_gap_seconds == 0:
            raise ValueError("max_crossing_gap_seconds harus lebih besar dari nol.")
        classes = tuple(self.vehicle_classes)
        if len(classes) != 4 or set(classes) != set(VEHICLE_CLASSES):
            raise ValueError("MVP memakai tepat empat kelas: car, motorcycle, bus, truck.")
        object.__setattr__(self, "vehicle_classes", classes)
        if not isinstance(self.preprocess, PreprocessConfig):
            raise ValueError("preprocess harus berupa PreprocessConfig.")
        if not isinstance(self.counting_line, CountingLineConfig):
            raise ValueError("counting_line harus berupa CountingLineConfig.")


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    source_name: str
    width: int
    height: int
    source_fps: float
    frame_count: int | None = None
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        _nonempty(self.source_name, "source_name")
        _integer(self.width, "width", 2)
        _integer(self.height, "height", 2)
        _finite(self.source_fps, "source_fps", 0)
        if self.source_fps == 0:
            raise ValueError("FPS sumber harus lebih besar dari nol.")
        if self.frame_count is not None:
            _integer(self.frame_count, "frame_count")
        if self.duration_seconds is not None:
            _finite(self.duration_seconds, "duration_seconds", 0)


@dataclass(frozen=True, slots=True)
class TrackObservation:
    frame_index: int
    timestamp_seconds: float
    track_id: int
    class_name: str
    confidence: float
    bbox_xyxy: BBox

    def __post_init__(self) -> None:
        _integer(self.frame_index, "frame_index")
        _integer(self.track_id, "track_id")
        _finite(self.timestamp_seconds, "timestamp_seconds", 0)
        _nonempty(self.class_name, "class_name")
        _finite(self.confidence, "confidence", 0)
        if self.confidence > 1:
            raise ValueError("confidence harus <= 1.")
        box = tuple(self.bbox_xyxy)
        if len(box) != 4:
            raise ValueError("bbox_xyxy memerlukan empat koordinat.")
        for value in box:
            _finite(value, "bbox_xyxy")
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("Bounding box harus memiliki lebar dan tinggi positif.")
        object.__setattr__(self, "bbox_xyxy", box)

    @property
    def anchor(self) -> Point:
        x1, _, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2, y2)


@dataclass(frozen=True, slots=True)
class CountEvent:
    run_id: str
    line_id: str
    track_id: int
    frame_index: int
    timestamp_seconds: float
    class_name: str
    direction: str


@dataclass(frozen=True, slots=True)
class CountSummary:
    total: int
    by_class_direction: dict[str, dict[str, int]]


@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    run_id: str
    processed_frames: int
    total_frames: int | None
    elapsed_seconds: float
    summary: CountSummary
    preview_frame_bgr: NDArray[np.uint8] | None = None


@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: str
    status: str
    summary: CountSummary
    output_dir: str
    artifact_paths: dict[str, str]
    warnings: list[str]
