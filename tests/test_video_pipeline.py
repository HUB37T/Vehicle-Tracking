import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("imageio_ffmpeg")

from vehicle_counter.contracts import (  # noqa: E402
    PreprocessConfig,
    RunConfig,
    TrackObservation,
    VideoMetadata,
)
from vehicle_counter.pipeline import process_video  # noqa: E402
from vehicle_counter.video import (  # noqa: E402
    AnnotatedVideoWriter,
    VideoReader,
    inspect_video,
    verify_output_video,
)


class SyntheticTracker:
    metadata = {"kind": "synthetic-observations", "is_model_inference": False}
    warnings = []

    def __init__(self, config, metadata):
        self.frames = []

    def track_frame(self, frame_bgr, *, frame_index, timestamp_seconds):
        self.frames.append(frame_bgr.copy())
        y = 30 + frame_index * 6
        return [
            TrackObservation(
                frame_index,
                timestamp_seconds,
                1,
                "car",
                0.9,
                (70, y - 15, 90, y),
            )
        ]


@pytest.fixture
def synthetic_video(tmp_path):
    path = tmp_path / "synthetic.mp4"
    metadata = VideoMetadata(path.name, 160, 120, 10, 12, 1.2)
    ramp = np.tile(np.linspace(20, 70, 160, dtype=np.uint8), (120, 1))
    frame = np.stack([ramp, ramp, ramp], axis=-1)
    with AnnotatedVideoWriter(path, metadata) as writer:
        for _ in range(12):
            writer.write(frame)
    return path


def test_complete_pipeline_artifacts_match_events_and_source(synthetic_video, tmp_path):
    updates = []

    def on_progress(update):
        updates.append(update)
        update.summary.by_class_direction["car"]["A_TO_B"] = 999
        if update.preview_frame_bgr is not None:
            update.preview_frame_bgr[:] = 255

    result = process_video(
        synthetic_video,
        RunConfig(),
        tmp_path / "outputs",
        on_progress,
        tracker_factory=SyntheticTracker,
    )
    assert result.status == "completed"
    assert result.summary.total == result.summary.by_class_direction["car"]["A_TO_B"] == 1
    manifest = json.loads(Path(result.artifact_paths["run_json"]).read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["processed_frames"] == 12
    assert manifest["tracker"]["is_model_inference"] is False
    assert manifest["processing_fps"] > 0
    assert manifest["summary"]["total"] == 1
    assert manifest["process_evidence"] == {
        "frame_index": 8,
        "timestamp_seconds": 0.8,
        "selection": "first_crossing",
        "clahe_active": False,
        "detections": None,
        "tracks": 1,
    }
    evidence_path = Path(result.artifact_paths["process_evidence"])
    assert evidence_path.name == "process_evidence.png"
    assert evidence_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert len(manifest["source"]["sha256"]) == 64
    assert updates[-1].processed_frames == 12
    assert len({update.run_id for update in updates}) == 1
    with Path(result.artifact_paths["events_csv"]).open(newline="", encoding="utf-8") as handle:
        events = list(csv.DictReader(handle))
    with Path(result.artifact_paths["counts_csv"]).open(newline="", encoding="utf-8") as handle:
        counts = list(csv.DictReader(handle))
    assert len(events) == 1
    assert events[0]["direction"] == "A_TO_B"
    assert events[0]["frame_index"] == "8"
    assert events[0]["timestamp_seconds"] == "0.8"
    assert len(counts) == 8
    assert sum(int(row["count"]) for row in counts) == len(events)
    verify_output_video(
        Path(result.artifact_paths["annotated_video"]), inspect_video(synthetic_video), 12
    )
    with VideoReader(Path(result.artifact_paths["annotated_video"])) as output:
        assert not np.all(next(iter(output))[1] == 255)


def test_processing_again_uses_new_run_and_fresh_counter(synthetic_video, tmp_path):
    results = [
        process_video(synthetic_video, RunConfig(), tmp_path, tracker_factory=SyntheticTracker)
        for _ in range(2)
    ]
    assert all(result.status == "completed" and result.summary.total == 1 for result in results)
    assert results[0].run_id != results[1].run_id


def test_zero_detections_still_exports_valid_empty_events(synthetic_video, tmp_path):
    class EmptyTracker(SyntheticTracker):
        def track_frame(self, frame_bgr, **kwargs):
            return []

    result = process_video(synthetic_video, RunConfig(), tmp_path, tracker_factory=EmptyTracker)
    assert result.status == "completed"
    assert result.summary.total == 0
    with Path(result.artifact_paths["events_csv"]).open(encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == []
    with Path(result.artifact_paths["counts_csv"]).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert all(row["count"] == "0" for row in rows)


def test_inference_failure_marks_run_failed_and_preserves_partial_counts(synthetic_video, tmp_path):
    class FailingTracker(SyntheticTracker):
        def track_frame(self, frame_bgr, **kwargs):
            if kwargs["frame_index"] == 9:
                raise RuntimeError("fixture inference failure")
            return super().track_frame(frame_bgr, **kwargs)

    result = process_video(synthetic_video, RunConfig(), tmp_path, tracker_factory=FailingTracker)
    assert result.status == "failed"
    manifest = json.loads(Path(result.artifact_paths["run_json"]).read_text())
    assert manifest["processed_frames"] == 9
    assert manifest["error"]["message"] == "fixture inference failure"
    assert result.summary.total == 1
    assert "annotated_video" not in result.artifact_paths
    assert inspect_video(synthetic_video).frame_count == 12


def test_corrupt_input_fails_before_run_creation(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a video")
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        process_video(path, RunConfig(), output, tracker_factory=SyntheticTracker)
    assert not output.exists()


def test_clahe_affects_inference_but_video_annotation_uses_source(synthetic_video, tmp_path):
    trackers = []

    def factory(config, metadata):
        tracker = SyntheticTracker(config, metadata)
        trackers.append(tracker)
        return tracker

    result = process_video(
        synthetic_video,
        replace(RunConfig(), preprocess=PreprocessConfig(mode="clahe")),
        tmp_path,
        tracker_factory=factory,
    )
    assert result.status == "completed"
    with (
        VideoReader(synthetic_video) as source,
        VideoReader(Path(result.artifact_paths["annotated_video"])) as output,
    ):
        original = next(iter(source))[1]
        annotated = next(iter(output))[1]
    assert not np.array_equal(trackers[0].frames[0], original)
    np.testing.assert_allclose(annotated[110, 150], original[110, 150], atol=5)


def test_failed_output_verification_prevents_completed_status(
    monkeypatch, synthetic_video, tmp_path
):
    def fail(*args):
        raise RuntimeError("output verification failed")

    monkeypatch.setattr("vehicle_counter.pipeline.verify_output_video", fail)
    result = process_video(synthetic_video, RunConfig(), tmp_path, tracker_factory=SyntheticTracker)
    assert result.status == "failed"
    assert "annotated_video" not in result.artifact_paths


def test_reader_validates_declared_frame_count_and_releases_capture(monkeypatch, tmp_path):
    import cv2

    class TruncatedCapture:
        released = False
        reads = 0

        def isOpened(self):
            return True

        def read(self):
            self.reads += 1
            return (
                (True, np.zeros((120, 160, 3), dtype=np.uint8))
                if self.reads <= 2
                else (False, None)
            )

        def get(self, key):
            return {cv2.CAP_PROP_FPS: 10, cv2.CAP_PROP_FRAME_COUNT: 5}[key]

        def release(self):
            self.released = True

    path = tmp_path / "partial.mp4"
    path.touch()
    capture = TruncatedCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    with pytest.raises(ValueError, match="terpotong"):
        with VideoReader(path) as reader:
            list(reader)
    assert capture.released


def test_invalid_fps_releases_open_capture(monkeypatch, tmp_path):
    import cv2

    class InvalidFpsCapture:
        released = False

        def isOpened(self):
            return True

        def read(self):
            return True, np.zeros((120, 160, 3), dtype=np.uint8)

        def get(self, key):
            return 12 if key == cv2.CAP_PROP_FRAME_COUNT else 0

        def release(self):
            self.released = True

    path = tmp_path / "invalid-fps.mp4"
    path.touch()
    capture = InvalidFpsCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    with pytest.raises(ValueError, match="FPS"):
        VideoReader(path)
    assert capture.released


def test_progress_callback_error_does_not_publish_success(synthetic_video, tmp_path):
    def on_progress(update):
        raise RuntimeError("fixture UI callback failure")

    result = process_video(
        synthetic_video, RunConfig(), tmp_path, on_progress, tracker_factory=SyntheticTracker
    )
    assert result.status == "failed"
    manifest = json.loads(Path(result.artifact_paths["run_json"]).read_text())
    assert manifest["error"]["message"] == "fixture UI callback failure"
    assert manifest["processed_frames"] == 1


def test_pcd_lab_exports_both_panels_without_reporting_motion_in_static_scene(tmp_path):
    import cv2

    from vehicle_counter.pcd_lab import export_lab

    path = tmp_path / "static.mp4"
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[30:65, 50:90] = 255
    with AnnotatedVideoWriter(path, VideoMetadata(path.name, 160, 120, 10)) as writer:
        writer.write(frame)
        writer.write(frame)
    output = export_lab(path, tmp_path, PreprocessConfig(), frame_index=1, min_area=100)
    directory = Path(output["output_dir"])
    report = json.loads((directory / "lab.json").read_text())
    assert report["reference_frame_index"] == 0
    assert report["frame_index"] == 1
    assert report["motion_components"] == []
    assert report["flow_arrows"] == 0
    for name in ("classical.png", "preprocessing.png"):
        assert cv2.imdecode(np.frombuffer((directory / name).read_bytes(), np.uint8), 1) is not None
