"""Shared data contracts; image coordinates always refer to the original frame."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Real

import numpy as np
from numpy.typing import NDArray

VEHICLE_CLASSES = ("car", "motorcycle", "bus", "truck")
DIRECTIONS = ("A_TO_B", "B_TO_A")
PREPROCESS_MODES = ("none", "clahe", "gamma", "gaussian", "unsharp", "median", "bilateral_clahe")
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
    gamma: float = 0.75
    gaussian_kernel_size: int = 5
    unsharp_amount: float = 0.7
    bilateral_diameter: int = 5
    bilateral_sigma_color: float = 35.0
    bilateral_sigma_space: float = 35.0

    def __post_init__(self) -> None:
        if self.mode not in PREPROCESS_MODES:
            raise ValueError(
                f"Mode prapemrosesan harus salah satu dari {', '.join(PREPROCESS_MODES)}."
            )
        _finite(self.gamma, "gamma", 0)
        if self.gamma == 0:
            raise ValueError("gamma harus lebih besar dari nol.")
        _integer(self.gaussian_kernel_size, "gaussian_kernel_size", 3)
        if self.gaussian_kernel_size > 31 or self.gaussian_kernel_size % 2 == 0:
            raise ValueError("gaussian_kernel_size harus ganjil pada rentang 3–31.")
        _finite(self.unsharp_amount, "unsharp_amount", 0)
        if self.unsharp_amount > 5:
            raise ValueError("unsharp_amount harus berada pada rentang 0–5.")
        _finite(self.clahe_clip_limit, "clahe_clip_limit", 0)
        if self.clahe_clip_limit == 0:
            raise ValueError("clahe_clip_limit harus lebih besar dari nol.")
        tiles = tuple(self.clahe_tile_grid_size)
        if len(tiles) != 2:
            raise ValueError("clahe_tile_grid_size memerlukan dua bilangan bulat.")
        for value in tiles:
            _integer(value, "clahe_tile_grid_size", 1)
        object.__setattr__(self, "clahe_tile_grid_size", tiles)
        _integer(self.bilateral_diameter, "bilateral_diameter", 3)
        if self.bilateral_diameter > 15 or self.bilateral_diameter % 2 == 0:
            raise ValueError("bilateral_diameter harus ganjil pada rentang 3–15.")
        for name in ("bilateral_sigma_color", "bilateral_sigma_space"):
            _finite(getattr(self, name), name, 0)
            if not 0 < getattr(self, name) <= 255:
                raise ValueError(f"{name} harus pada rentang (0, 255].")


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
class CongestionConfig:
    enabled: bool = False
    roi: BBox = (0.05, 0.45, 0.95, 0.95)
    min_vehicles: int = 3
    min_occupancy: float = 0.15
    slow_speed: float = 0.15
    min_slow_fraction: float = 0.65
    motion_window_seconds: float = 2.0
    min_motion_seconds: float = 1.0
    hold_seconds: float = 3.0
    recovery_seconds: float = 2.0
    sample_seconds: float = 0.5
    dropout_grace_seconds: float = 0.0

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("congestion.enabled harus boolean.")
        roi = tuple(self.roi)
        if len(roi) != 4:
            raise ValueError("congestion.roi memerlukan empat koordinat.")
        for value in roi:
            _finite(value, "congestion.roi", 0)
        if not (roi[0] < roi[2] <= 1 and roi[1] < roi[3] <= 1):
            raise ValueError("congestion.roi harus terurut pada rentang 0-1.")
        object.__setattr__(self, "roi", roi)
        _integer(self.min_vehicles, "min_vehicles", 2)
        for name in ("min_occupancy", "min_slow_fraction"):
            value = getattr(self, name)
            _finite(value, name, 0)
            if not 0 < value <= 1:
                raise ValueError(f"{name} harus berada pada rentang (0, 1].")
        for name in (
            "slow_speed",
            "motion_window_seconds",
            "min_motion_seconds",
            "hold_seconds",
            "recovery_seconds",
            "sample_seconds",
        ):
            value = getattr(self, name)
            _finite(value, name, 0)
            if value <= 0:
                raise ValueError(f"{name} harus lebih besar dari nol.")
        if self.min_motion_seconds > self.motion_window_seconds:
            raise ValueError("min_motion_seconds melebihi motion_window_seconds.")
        _finite(self.dropout_grace_seconds, "dropout_grace_seconds", 0)
        if self.dropout_grace_seconds > 0.5:
            raise ValueError("dropout_grace_seconds maksimal 0,5 detik.")


@dataclass(frozen=True, slots=True)
class TrafficZoneConfig:
    zone_id: str = "zone-1"
    name: str = "Lajur 1"
    polygon: tuple[Point, ...] = ((0.05, 0.45), (0.48, 0.45), (0.48, 0.95), (0.05, 0.95))

    def __post_init__(self):
        from vehicle_counter.polygons import validated_polygon

        for name in ("zone_id", "name"):
            _nonempty(getattr(self, name), name)
            if len(getattr(self, name)) > 80:
                raise ValueError(f"{name} terlalu panjang.")
        if not all(c.isalnum() or c in "-_" for c in self.zone_id):
            raise ValueError("zone_id hanya boleh huruf, angka, '-' dan '_'.")
        object.__setattr__(self, "polygon", validated_polygon(self.polygon))


@dataclass(frozen=True, slots=True)
class RunConfig:
    model_weights: str = "yolo11n.pt"
    tracker_config: str = "bytetrack.yaml"
    device: str = "cpu"
    imgsz: int = 640
    detector_conf: float = 0.10
    detector_deduplicate_classes: bool = False
    vehicle_classes: tuple[str, ...] = VEHICLE_CLASSES
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    counting_line: CountingLineConfig = field(default_factory=CountingLineConfig)
    min_track_hits: int = 3
    max_crossing_gap_seconds: float = 0.5
    deduplicate_crossings: bool = False
    guard_dense_crossings: bool = False
    truck_review_weights: str | None = None
    truck_review_light_threshold: float = 0.40
    truck_review_minivan_only: bool = False
    detection_roi: BBox | None = None
    congestion: CongestionConfig = field(default_factory=CongestionConfig)
    traffic_zones: tuple[TrafficZoneConfig, ...] = ()
    heatmap_enabled: bool = False
    quality_check: bool = True

    def __post_init__(self) -> None:
        for name in ("model_weights", "tracker_config", "device"):
            _nonempty(getattr(self, name), name)
        if type(self.deduplicate_crossings) is not bool:
            raise ValueError("deduplicate_crossings harus boolean.")
        if type(self.guard_dense_crossings) is not bool:
            raise ValueError("guard_dense_crossings harus boolean.")
        if type(self.detector_deduplicate_classes) is not bool:
            raise ValueError("detector_deduplicate_classes harus boolean.")
        if self.truck_review_weights is not None:
            _nonempty(self.truck_review_weights, "truck_review_weights")
        _finite(self.truck_review_light_threshold, "truck_review_light_threshold", 0)
        if not 0 < self.truck_review_light_threshold <= 1:
            raise ValueError("truck_review_light_threshold harus lebih dari 0 dan <= 1.")
        if type(self.truck_review_minivan_only) is not bool:
            raise ValueError("truck_review_minivan_only harus boolean.")
        if self.detection_roi is not None:
            roi = tuple(self.detection_roi)
            if len(roi) != 4:
                raise ValueError("detection_roi memerlukan empat koordinat [x1, y1, x2, y2].")
            for value in roi:
                _finite(value, "detection_roi", 0)
            if not (roi[0] < roi[2] <= 1 and roi[1] < roi[3] <= 1):
                raise ValueError("detection_roi harus terurut dan berada pada rentang [0, 1].")
            object.__setattr__(self, "detection_roi", roi)
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
        if not isinstance(self.congestion, CongestionConfig):
            raise ValueError("congestion harus berupa CongestionConfig.")
        zones = tuple(self.traffic_zones)
        if len(zones) > 4 or any(not isinstance(z, TrafficZoneConfig) for z in zones):
            raise ValueError("traffic_zones berisi maksimal empat TrafficZoneConfig.")
        if len({z.zone_id for z in zones}) != len(zones):
            raise ValueError("zone_id harus unik.")
        for name in ("heatmap_enabled", "quality_check"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} harus boolean.")
        object.__setattr__(self, "traffic_zones", zones)


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
