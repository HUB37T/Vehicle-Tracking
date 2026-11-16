import io
import json
import zipfile
from dataclasses import replace

import cv2
import numpy as np
import pytest

from vehicle_counter.config import config_from_dict, config_to_dict
from vehicle_counter.contracts import (
    CongestionConfig,
    PreprocessConfig,
    RunConfig,
    TrackObservation,
    TrafficZoneConfig,
    VideoMetadata,
)
from vehicle_counter.heatmap import PresenceHeatmap
from vehicle_counter.preprocessing import preprocess_frame
from vehicle_counter.quality import frame_metrics, inspect_quality
from vehicle_counter.traffic import TrafficMonitor
from vehicle_counter.training_labels import export_dataset, save_labels, validate_boxes
from vehicle_counter.video import AnnotatedVideoWriter


@pytest.mark.parametrize(
    "polygon",
    [
        ((0, 0), (1, 1), (1, 0), (0, 1)),
        ((0, 0), (0.2, 0), (0.4, 0)),
        ((0, 0), (1, 0), (1, 0), (0, 1)),
        ((-0.1, 0), (1, 0), (0, 1)),
    ],
)
def test_invalid_zones_rejected(polygon):
    with pytest.raises(ValueError):
        TrafficZoneConfig(polygon=polygon)


def test_zones_roundtrip_strict_config_and_count_anchor_in_polygon():
    zone = TrafficZoneConfig(polygon=((0, 0), (1, 0), (0, 1)))
    config = replace(RunConfig(), traffic_zones=(zone,), heatmap_enabled=True)
    assert config_from_dict(config_to_dict(config)) == config
    with pytest.raises(ValueError):
        replace(config, traffic_zones=(zone, zone))
    monitor = TrafficMonitor(
        CongestionConfig(enabled=True), width=200, height=200, max_gap=0.5, polygon=zone.polygon
    )
    inside = TrackObservation(0, 0, 1, "car", 0.9, (10, 5, 30, 30))
    outside = TrackObservation(0, 0, 2, "car", 0.9, (150, 140, 190, 180))
    monitor.update([inside, outside], frame_index=0, timestamp=0)
    assert monitor.current.vehicle_count == 1
    assert 0 < monitor.current.occupancy_fraction < 1


def test_heatmap_is_time_weighted_and_does_not_double_same_cell(tmp_path):
    track = TrackObservation(0, 0, 1, "car", 0.9, (10, 10, 20, 20))
    results = []
    for fps in (10, 20):
        heat = PresenceHeatmap(100, 100, fps)
        for _ in range(fps):
            heat.update([track, replace(track, track_id=2)])
        assert heat.seconds.max() == pytest.approx(1)
        results.append(heat.seconds)
    np.testing.assert_allclose(*results)
    artifacts = heat.export(tmp_path, np.zeros((100, 100, 3), np.uint8))
    with np.load(artifacts["heatmap_npz"], allow_pickle=False) as data:
        np.testing.assert_allclose(data["seconds"], heat.seconds)
    assert cv2.imread(artifacts["heatmap_png"]).shape == (100, 100, 3)


def test_quality_metrics_distinguish_intensity_and_detail_without_changing_input():
    rng = np.random.default_rng(7)
    gradient = np.tile(np.linspace(0, 170, 128, dtype=np.uint8), (128, 1))
    image = np.repeat(gradient[:, :, None], 3, axis=2) + rng.integers(
        0, 80, (128, 128, 3), dtype=np.uint8
    )
    original = image.copy()
    sharp, *_ = frame_metrics(image)
    blurred, *_ = frame_metrics(cv2.GaussianBlur(image, (9, 9), 2))
    assert sharp["laplacian_variance"] > blurred["laplacian_variance"]
    assert sharp["high_frequency_energy_fraction"] > blurred["high_frequency_energy_fraction"]
    assert frame_metrics(np.zeros_like(image))[0]["brightness_mean"] == 0
    np.testing.assert_array_equal(image, original)


def test_orb_background_motion_is_measured_on_translated_fixture(tmp_path):
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    source = tmp_path / "motion.mp4"
    with AnnotatedVideoWriter(source, VideoMetadata(source.name, 320, 240, 10, 31, 3.1)) as writer:
        for index in range(31):
            shifted = cv2.warpAffine(
                frame,
                np.float32([[1, 0, index], [0, 1, 0]]),
                (320, 240),
                borderMode=cv2.BORDER_REFLECT,
            )
            writer.write(shifted)
    report, panel = inspect_quality(source, RunConfig())
    assert report["background_motion"]["status"] == "candidate_camera_motion"
    assert len(report["background_motion"]["pairs"]) >= 2
    assert panel.ndim == 3 and report["thresholds_are_heuristic"] is True


def test_median_removes_impulse_and_bilateral_clahe_preserves_contract():
    image = np.full((32, 32, 3), 80, np.uint8)
    image[16, 16] = 255
    assert np.all(preprocess_frame(image, PreprocessConfig(mode="median"))[16, 16] == 80)
    output = preprocess_frame(image, PreprocessConfig(mode="bilateral_clahe"))
    assert output.dtype == np.uint8 and output.shape == image.shape
    assert np.all(image[16, 16] == 255)


def test_manual_labels_export_latest_revision_and_original_image(tmp_path):
    root = tmp_path / "labels"
    frame = np.zeros((80, 100, 3), np.uint8)
    frame[:, :, 1] = 120
    common = {"source_sha256": "a" * 64, "frame_index": 7, "frame_bgr": frame, "confirmed": True}
    save_labels(root, boxes=[{"class_name": "truck", "bbox": [0.1, 0.2, 0.5, 0.8]}], **common)
    save_labels(root, boxes=[{"class_name": "car", "bbox": [0.1, 0.2, 0.5, 0.8]}], **common)
    with zipfile.ZipFile(io.BytesIO(export_dataset(root))) as archive:
        labels = [p for p in archive.namelist() if p.startswith("labels/")]
        assert len(labels) == 1
        assert archive.read(labels[0]).decode().split() == [
            "0",
            "0.30000000",
            "0.50000000",
            "0.40000000",
            "0.60000000",
        ]
        name = next(p for p in archive.namelist() if p.startswith("images/"))
        decoded = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(decoded, frame)
        assert json.loads(archive.read("metadata.json"))["split_status"] == "not_assigned"
    assert len(list(root.glob("*/annotation.json"))) == 2
    with pytest.raises(ValueError, match="konfirmasi"):
        save_labels(root, **{**common, "confirmed": False}, boxes=[])
    with pytest.raises(ValueError):
        validate_boxes([{"class_name": "car", "bbox": [0.8, 0.2, 0.1, 0.6]}])


def test_two_lane_monitors_do_not_mix_stationary_and_moving_tracks():
    config = CongestionConfig(enabled=True, min_occupancy=0.05)
    left = TrafficMonitor(
        config, width=600, height=200, max_gap=0.5, polygon=((0, 0), (0.5, 0), (0.5, 1), (0, 1))
    )
    right = TrafficMonitor(
        config, width=600, height=200, max_gap=0.5, polygon=((0.5, 0), (1, 0), (1, 1), (0.5, 1))
    )
    for frame in range(46):
        observations = [
            TrackObservation(frame, frame / 10, i, "car", 0.9, (10 + i * 70, 30, 50 + i * 70, 70))
            for i in range(3)
        ]
        observations += [
            TrackObservation(
                frame,
                frame / 10,
                i + 3,
                "car",
                0.9,
                (310 + i * 70 + frame, 30, 350 + i * 70 + frame, 70),
            )
            for i in range(3)
        ]
        for monitor in (left, right):
            monitor.update(observations, frame_index=frame, timestamp=frame / 10)
    assert left.current.status == "congested" and left.current.vehicle_count == 3
    assert right.current.status == "dense" and right.current.vehicle_count == 3
