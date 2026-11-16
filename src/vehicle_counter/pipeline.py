"""Sequential video analysis and run lifecycle for CLI and future Streamlit UI."""

from __future__ import annotations

import platform
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from itertools import islice
from math import ceil, isfinite
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from vehicle_counter.config import config_to_dict
from vehicle_counter.contracts import ProgressUpdate, RunConfig, RunResult, VideoMetadata
from vehicle_counter.counting import VehicleCounter
from vehicle_counter.crossing_audit import audit_crossings, write_crossing_audit
from vehicle_counter.execution import AnalysisCancelled, check_cancel
from vehicle_counter.exporting import export_counts, file_sha256, write_json
from vehicle_counter.heatmap import PresenceHeatmap
from vehicle_counter.preprocessing import preprocess_frame
from vehicle_counter.run_analysis import CrossingDiagnostics, export_analysis, export_zone_analysis
from vehicle_counter.run_evidence import render_evidence, write_evidence
from vehicle_counter.tracking import FrameTracker, YoloTracker
from vehicle_counter.traffic import TrafficMonitor
from vehicle_counter.truck_review import TruckCropReviewer
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
        "deep-sort-realtime",
        "scipy",
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
    max_seconds: float | None = None,
    source_label: str | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> RunResult:
    """Process one local CFR MP4 with fresh tracker/counter state.

    Invalid video inputs raise before run creation. Errors after creation return
    status='failed', with details in run.json and warnings. Interrupts also mark
    the run failed before propagating. An injected factory is a testing seam;
    its metadata must honestly distinguish fake tracks from model inference.
    """
    if max_seconds is not None and (
        isinstance(max_seconds, bool)
        or not isinstance(max_seconds, (int, float))
        or not isfinite(max_seconds)
        or max_seconds <= 0
    ):
        raise ValueError("max_seconds harus angka positif yang berhingga.")
    if source_label is not None and (not isinstance(source_label, str) or not source_label.strip()):
        raise ValueError("Nama tampilan sumber harus teks yang tidak kosong.")
    total_start = perf_counter()
    with VideoReader(source_path) as reader:
        metadata = reader.metadata
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:12]
        output_dir = Path(output_root).expanduser().resolve() / run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = output_dir / "run.json"
        video_path = output_dir / "annotated.mp4"
        counter = VehicleCounter(run_id, config, width=metadata.width, height=metadata.height)
        diagnostics = CrossingDiagnostics(counter.line, max_gap=config.max_crossing_gap_seconds)
        traffic = (
            TrafficMonitor(
                config.congestion,
                width=metadata.width,
                height=metadata.height,
                max_gap=config.max_crossing_gap_seconds,
            )
            if config.congestion.enabled
            else None
        )
        zones = (
            [
                (
                    zone,
                    TrafficMonitor(
                        config.congestion,
                        width=metadata.width,
                        height=metadata.height,
                        max_gap=config.max_crossing_gap_seconds,
                        polygon=zone.polygon,
                    ),
                )
                for zone in config.traffic_zones
            ]
            if config.congestion.enabled
            else []
        )
        heatmap = (
            PresenceHeatmap(metadata.width, metadata.height, metadata.source_fps)
            if config.heatmap_enabled
            else None
        )
        heatmap_background = None
        frame_limit = ceil(max_seconds * metadata.source_fps) if max_seconds else None
        progress_total = (
            min(metadata.frame_count, frame_limit)
            if (metadata.frame_count is not None and frame_limit is not None)
            else (frame_limit or metadata.frame_count)
        )
        annotator = FrameAnnotator(counter.line)
        warnings = [
            "Timestamp mengasumsikan video constant frame rate (CFR). Konversi VFR dahulu.",
            "Video beranotasi tidak mempertahankan audio sumber.",
        ]
        if max_seconds is not None:
            warnings.append(f"Uji singkat memproses paling lama {max_seconds:g} detik awal video.")
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
            "video_metadata": {
                **asdict(metadata),
                "source_name": source_label or metadata.source_name,
            },
            "config": config_to_dict(config),
            "analysis_scope": {
                "kind": "preview" if max_seconds is not None else "full",
                "max_seconds": max_seconds,
            },
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
        audit_rows = []
        evidence = None
        evidence_priority = -1
        try:
            check_cancel(should_cancel)
            manifest["source"]["sha256"] = file_sha256(reader.path)
            if config.quality_check:
                from vehicle_counter.quality import export_quality, inspect_quality

                quality, panel = inspect_quality(reader.path, config, should_cancel=should_cancel)
                manifest["quality"] = quality
                artifacts.update(export_quality(output_dir, quality, panel))
                warnings.extend(quality["warnings"])
            check_cancel(should_cancel)
            model_start = perf_counter()
            tracker = (tracker_factory or YoloTracker)(config, metadata)
            review_truck = (
                TruckCropReviewer(config.truck_review_weights)
                if config.truck_review_weights
                else None
            )
            manifest["model_load_seconds"] = perf_counter() - model_start
            manifest["tracker"] = tracker.metadata
            if review_truck is not None:
                manifest["truck_review"] = review_truck.metadata
            warnings.extend(tracker.warnings)
            if not tracker.metadata.get("is_model_inference", False):
                warnings.append(
                    "Tracker disuntikkan untuk pengujian; hasil bukan inferensi model nyata."
                )
            write_json(manifest_path, manifest)
            preview_interval = max(1, round(metadata.source_fps / 2))
            processing_start = perf_counter()
            with AnnotatedVideoWriter(video_path, metadata) as writer:
                frames = islice(reader, frame_limit) if frame_limit is not None else iter(reader)
                for frame_index, frame in frames:
                    check_cancel(should_cancel)
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
                    events = counter.update(
                        observations, frame_bgr=frame, review_truck=review_truck
                    )
                    detections = getattr(tracker, "last_detections", None)
                    diagnostics.update(observations, detections)
                    if traffic is not None:
                        traffic.update(observations, frame_index=frame_index, timestamp=timestamp)
                    for _, monitor in zones:
                        monitor.update(observations, frame_index=frame_index, timestamp=timestamp)
                    if heatmap is not None:
                        heatmap.update(observations)
                        if heatmap_background is None:
                            import cv2

                            scale = min(1, 1280 / max(frame.shape[:2]))
                            heatmap_background = cv2.resize(
                                frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                            )
                    audit_rows.extend(audit_crossings(events, observations, detections))
                    annotated = annotator.render(
                        frame,
                        observations,
                        counter.summary(),
                        timestamp_seconds=timestamp,
                    )
                    if traffic is not None:
                        annotated = traffic.draw_overlay(annotated)
                    if zones:
                        import cv2
                        import numpy as np

                        from vehicle_counter.traffic import TRAFFIC_LABELS

                        for number, (_, monitor) in enumerate(zones, 1):
                            color = (
                                (50, 70, 255)
                                if monitor.current.status == "congested"
                                else (80, 220, 160)
                            )
                            points = np.rint(monitor.polygon).astype(np.int32)
                            cv2.polylines(annotated, [points], True, color, 2)
                            label = f"Z{number}: {TRAFFIC_LABELS[monitor.current.status]}"
                            x, y = map(int, points[0])
                            cv2.putText(
                                annotated,
                                label,
                                (max(4, x), max(20, y - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2,
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
                                detection_roi_pixels=getattr(tracker, "detection_roi_pixels", None),
                                tracker_name=tracker.metadata.get("tracker_name", "ByteTrack"),
                            )
                            evidence_priority = priority
                            manifest["process_evidence"] = {
                                "frame_index": frame_index,
                                "timestamp_seconds": timestamp,
                                "selection": EVIDENCE_SELECTIONS[priority],
                                "clahe_active": config.preprocess.mode
                                in ("clahe", "bilateral_clahe"),
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
                                total_frames=progress_total,
                                elapsed_seconds=perf_counter() - processing_start,
                                summary=counter.summary(),
                                preview_frame_bgr=annotated.copy(),
                            )
                        )
            check_cancel(should_cancel)
            verify_output_video(video_path, metadata, processed)
            artifacts["annotated_video"] = str(video_path)
            manifest["diagnostics"] = diagnostics.finish(len(counter.events))
            manifest["analyzed_duration_seconds"] = processed / metadata.source_fps
            manifest["traffic"] = (
                traffic.summary(processed / metadata.source_fps) if traffic is not None else None
            )
            manifest["traffic_zones"] = [
                {
                    "zone_id": zone.zone_id,
                    "name": zone.name,
                    "polygon": zone.polygon,
                    **monitor.summary(processed / metadata.source_fps),
                }
                for zone, monitor in zones
            ]
            if zones:
                artifacts.update(
                    export_zone_analysis(output_dir, zones, processed / metadata.source_fps)
                )
            if heatmap is not None and heatmap_background is not None:
                artifacts.update(heatmap.export(output_dir, heatmap_background))
            artifacts.update(
                export_analysis(
                    output_dir,
                    counter.events,
                    processed / metadata.source_fps,
                    manifest["diagnostics"],
                    traffic,
                )
            )
            audit_path = output_dir / "crossing_audit.csv"
            write_crossing_audit(audit_path, audit_rows)
            artifacts["crossing_audit"] = str(audit_path)
            manifest["crossing_audit"] = {
                "events": len(audit_rows),
                "flagged_events": sum(bool(row["review_reason"]) for row in audit_rows),
                "interpretation": (
                    "Review candidates only; absence of flags does not verify identity."
                ),
            }
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
                        total_frames=progress_total,
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
            manifest["status"] = "cancelled" if isinstance(error, AnalysisCancelled) else "failed"
            manifest["process_evidence"] = None
            manifest["error"] = {"type": type(error).__name__, "message": str(error)}
            warnings.append(
                str(error) if isinstance(error, AnalysisCancelled) else f"Analisis gagal: {error}"
            )
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
