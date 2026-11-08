"""Sequential video analysis and run lifecycle for CLI and future Streamlit UI."""

from __future__ import annotations

import platform
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from vehicle_counter.config import config_to_dict
from vehicle_counter.contracts import ProgressUpdate, RunConfig, RunResult, VideoMetadata
from vehicle_counter.counting import VehicleCounter
from vehicle_counter.exporting import export_counts, file_sha256, write_json
from vehicle_counter.preprocessing import preprocess_frame
from vehicle_counter.run_evidence import render_evidence, write_evidence
from vehicle_counter.tracking import FrameTracker, YoloByteTracker
from vehicle_counter.video import AnnotatedVideoWriter, VideoReader, verify_output_video
from vehicle_counter.visualization import FrameAnnotator

TrackerFactory = Callable[[RunConfig, VideoMetadata], FrameTracker]
ProgressCallback = Callable[[ProgressUpdate], None]
EVIDENCE_SELECTIONS = ("first_frame", "vehicle_visible", "track_with_trail", "first_crossing")


def _versions() -> dict[str, str | None]:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in (
        "vehicle-counter",
        "numpy",
        "PyYAML",
        "opencv-python",
        "ultralytics",
        "torch",
        "torchvision",
        "lap",
        "imageio-ffmpeg",
    ):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def process_video(
    source_path: str | Path,
    config: RunConfig,
    output_root: str | Path = "outputs",
    on_progress: ProgressCallback | None = None,
    *,
    tracker_factory: TrackerFactory | None = None,
) -> RunResult:
    """Process one local CFR MP4 with fresh tracker/counter state.

    Invalid video inputs raise before run creation. Errors after creation return
    status='failed', with details in run.json and warnings. Interrupts also mark
    the run failed before propagating. An injected factory is a testing seam;
    its metadata must honestly distinguish fake tracks from model inference.
    """
    total_start = perf_counter()
    with VideoReader(source_path) as reader:
        metadata = reader.metadata
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:12]
        output_dir = Path(output_root).expanduser().resolve() / run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = output_dir / "run.json"
        video_path = output_dir / "annotated.mp4"
        counter = VehicleCounter(run_id, config, width=metadata.width, height=metadata.height)
        annotator = FrameAnnotator(counter.line)
        warnings = [
            "Timestamp mengasumsikan video constant frame rate (CFR). Konversi VFR dahulu.",
            "Video beranotasi tidak mempertahankan audio sumber.",
        ]
        if metadata.frame_count is None:
            warnings.append(
                "Jumlah frame sumber tidak diketahui; kelengkapan EOF tidak dapat dipastikan."
            )
        artifacts = {"run_json": str(manifest_path)}
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "running",
            "started_at_utc": datetime.now(UTC).isoformat(),
            "source": {"path": str(reader.path), "sha256": None, "kind": "local_video"},
            "video_metadata": asdict(metadata),
            "config": config_to_dict(config),
            "software": _versions(),
            "tracker": None,
            "process_evidence": None,
            "model_load_seconds": None,
            "processed_frames": 0,
            "processing_seconds": None,
            "processing_fps": None,
            "total_wall_seconds": None,
            "summary": asdict(counter.summary()),
            "warnings": warnings,
            "artifact_paths": artifacts,
            "error": None,
        }
        write_json(manifest_path, manifest)
        processed = 0
        processing_start = None
        last_preview = None
        evidence = None
        evidence_priority = -1
        try:
            manifest["source"]["sha256"] = file_sha256(reader.path)
            model_start = perf_counter()
            tracker = (tracker_factory or YoloByteTracker)(config, metadata)
            manifest["model_load_seconds"] = perf_counter() - model_start
            manifest["tracker"] = tracker.metadata
            warnings.extend(tracker.warnings)
            if not tracker.metadata.get("is_model_inference", False):
                warnings.append(
                    "Tracker disuntikkan untuk pengujian; hasil bukan inferensi model nyata."
                )
            write_json(manifest_path, manifest)
            preview_interval = max(1, round(metadata.source_fps / 2))
            processing_start = perf_counter()
            with AnnotatedVideoWriter(video_path, metadata) as writer:
                for frame_index, frame in reader:
                    timestamp = frame_index / metadata.source_fps
                    input_frame = preprocess_frame(frame, config.preprocess)
                    observations = tracker.track_frame(
                        input_frame,
                        frame_index=frame_index,
                        timestamp_seconds=timestamp,
                    )
                    if any(
                        item.frame_index != frame_index or item.timestamp_seconds != timestamp
                        for item in observations
                    ):
                        raise ValueError("Observasi tracker tidak sesuai frame/timestamp pipeline.")
                    events = counter.update(observations)
                    annotated = annotator.render(
                        frame,
                        observations,
                        counter.summary(),
                        timestamp_seconds=timestamp,
                    )
                    writer.write(annotated)
                    if evidence_priority < 3:
                        detections = getattr(tracker, "last_detections", None)
                        has_trail = any(
                            len(annotator.trail_for(item.track_id)) >= 2 for item in observations
                        )
                        priority = int(bool(observations or detections))
                        if has_trail:
                            priority = 2
                        if events:
                            priority = 3
                        if priority > evidence_priority:
                            evidence = render_evidence(
                                frame,
                                input_frame,
                                detections,
                                observations,
                                annotator,
                                config.preprocess,
                                frame_index=frame_index,
                                timestamp_seconds=timestamp,
                            )
                            evidence_priority = priority
                            manifest["process_evidence"] = {
                                "frame_index": frame_index,
                                "timestamp_seconds": timestamp,
                                "selection": EVIDENCE_SELECTIONS[priority],
                                "clahe_active": config.preprocess.mode == "clahe",
                                "detections": None if detections is None else len(detections),
                                "tracks": len(observations),
                            }
                    processed += 1
                    last_preview = annotated
                    if on_progress and (processed == 1 or processed % preview_interval == 0):
                        on_progress(
                            ProgressUpdate(
                                run_id=run_id,
                                processed_frames=processed,
                                total_frames=metadata.frame_count,
                                elapsed_seconds=perf_counter() - processing_start,
                                summary=counter.summary(),
                                preview_frame_bgr=annotated.copy(),
                            )
                        )
            verify_output_video(video_path, metadata, processed)
            artifacts["annotated_video"] = str(video_path)
            artifacts.update(
                export_counts(
                    output_dir,
                    run_id=run_id,
                    line_id=config.counting_line.line_id,
                    events=counter.events,
                    summary=counter.summary(),
                )
            )
            if on_progress:
                on_progress(
                    ProgressUpdate(
                        run_id=run_id,
                        processed_frames=processed,
                        total_frames=metadata.frame_count,
                        elapsed_seconds=perf_counter() - processing_start,
                        summary=counter.summary(),
                        preview_frame_bgr=last_preview.copy() if last_preview is not None else None,
                    )
                )
            if evidence is not None:
                evidence_path = output_dir / "process_evidence.png"
                write_evidence(evidence_path, evidence)
                artifacts["process_evidence"] = str(evidence_path)
            manifest["status"] = "completed"
        except BaseException as error:
            manifest["status"] = "failed"
            manifest["process_evidence"] = None
            manifest["error"] = {"type": type(error).__name__, "message": str(error)}
            warnings.append(f"Analisis gagal: {error}")
            # Preserve inspectable partial counts, always under failed run metadata.
            try:
                artifacts.update(
                    export_counts(
                        output_dir,
                        run_id=run_id,
                        line_id=config.counting_line.line_id,
                        events=counter.events,
                        summary=counter.summary(),
                    )
                )
            except Exception as export_error:
                warnings.append(f"Ekspor hitungan parsial gagal: {export_error}")
            if not isinstance(error, Exception):
                manifest.update(
                    processed_frames=processed,
                    summary=asdict(counter.summary()),
                    total_wall_seconds=perf_counter() - total_start,
                )
                write_json(manifest_path, manifest)
                raise
        processing_seconds = (
            perf_counter() - processing_start if processing_start is not None else None
        )
        manifest.update(
            processed_frames=processed,
            summary=asdict(counter.summary()),
            processing_seconds=processing_seconds,
            processing_fps=(processed / processing_seconds if processing_seconds else None),
            total_wall_seconds=perf_counter() - total_start,
            finished_at_utc=datetime.now(UTC).isoformat(),
        )
        write_json(manifest_path, manifest)
        return RunResult(
            run_id=run_id,
            status=manifest["status"],
            summary=counter.summary(),
            output_dir=str(output_dir),
            artifact_paths=dict(artifacts),
            warnings=list(warnings),
        )
