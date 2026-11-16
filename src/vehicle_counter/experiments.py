"""Sequential, reproducible comparisons with one source and fixed counting settings."""

import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from vehicle_counter.contracts import RunConfig, RunResult
from vehicle_counter.execution import AnalysisCancelled, check_cancel
from vehicle_counter.exporting import file_sha256, write_json
from vehicle_counter.pipeline import process_video
from vehicle_counter.run_analysis import write_rows


def run_comparison(
    source: Path,
    config: RunConfig,
    output_root: Path,
    *,
    modes: tuple[str, ...],
    trackers: tuple[str, ...],
    weights: tuple[str, ...] | None = None,
    source_label: str | None = None,
    max_seconds: float | None = None,
    on_run=None,
    processor=None,
    should_cancel=None,
    on_progress=None,
) -> dict:
    if not modes or not trackers:
        raise ValueError("Pilih sedikitnya satu mode dan tracker.")
    processor = processor or process_video
    selected = [
        replace(
            config,
            preprocess=replace(config.preprocess, mode=mode),
            tracker_config=tracker,
            model_weights=weight,
        )
        for mode in dict.fromkeys(modes)
        for tracker in dict.fromkeys(trackers)
        for weight in dict.fromkeys(weights if weights is not None else (config.model_weights,))
    ]
    if not selected:
        raise ValueError("Pilih sedikitnya satu bobot model.")
    batch = output_root / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8])
    batch.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "source_sha256": file_sha256(source),
        "source_name": source_label or source.name,
        "max_seconds": max_seconds,
        "rows": [],
        "results": [],
        "output_dir": str(batch.resolve()),
    }
    write_json(batch / "comparison.json", manifest)
    was_cancelled = False
    try:
        for index, choice in enumerate(selected):
            check_cancel(should_cancel)
            if on_run:
                on_run(index, len(selected), choice)
            label_args = {"source_label": source_label} if source_label is not None else {}
            if should_cancel is not None:
                label_args["should_cancel"] = should_cancel
            if on_progress is not None:
                label_args["on_progress"] = on_progress
            result: RunResult = processor(
                source, choice, batch / "runs", max_seconds=max_seconds, **label_args
            )
            manifest["results"].append(asdict(result))
            info_path = Path(result.artifact_paths["run_json"])
            info = json.loads(info_path.read_text(encoding="utf-8"))
            if info["source"]["sha256"] != manifest["source_sha256"]:
                raise ValueError(
                    "Video berubah selama perbandingan. Jalankan kembali dengan berkas tetap."
                )
            manifest["rows"].append(
                {
                    "run_id": result.run_id,
                    "status": result.status,
                    "preprocess": choice.preprocess.mode,
                    "model": Path(choice.model_weights).name,
                    "imgsz": choice.imgsz,
                    "detector_conf": choice.detector_conf,
                    "weights_sha256": (info.get("tracker") or {}).get("weights_sha256"),
                    "tracker": info.get("tracker", {}).get("tracker_name", choice.tracker_config)
                    if info.get("tracker")
                    else choice.tracker_config,
                    "total": result.summary.total,
                    "processing_fps": info.get("processing_fps"),
                    "processing_seconds": info.get("processing_seconds"),
                    "processed_frames": info.get("processed_frames"),
                    "analysis_scope": "preview" if max_seconds is not None else "full",
                    "error": (info.get("error") or {}).get("message", ""),
                }
            )
            write_json(batch / "comparison.json", manifest)
            if result.status == "cancelled":
                was_cancelled = True
                break
    except AnalysisCancelled:
        was_cancelled = True
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        write_json(batch / "comparison.json", manifest)
        raise
    manifest["status"] = (
        "cancelled"
        if was_cancelled
        else "completed"
        if all(r["status"] == "completed" for r in manifest["rows"])
        else "partial_failure"
    )
    if was_cancelled:
        manifest["status"] = "cancelled"
    write_rows(
        batch / "comparison.csv",
        manifest["rows"],
        list(manifest["rows"][0]) if manifest["rows"] else ["run_id", "status"],
    )
    write_json(batch / "comparison.json", manifest)
    return manifest
