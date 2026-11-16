"""Reopen saved runs without inference or changes to their automatic counts."""

import json
from pathlib import Path

from vehicle_counter.contracts import CountSummary, RunResult


def list_runs(root: Path, *, limit: int = 100) -> list[dict]:
    if limit < 1:
        raise ValueError("Batas riwayat harus positif.")
    rows = []
    for path in root.rglob("run.json"):
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "run_id": info["run_id"],
                    "source": info["video_metadata"]["source_name"],
                    "started_at_utc": info.get("started_at_utc", ""),
                    "status": info["status"],
                    "scope": info.get("analysis_scope", {}).get("kind", "full"),
                    "model": Path(info["config"]["model_weights"]).name,
                    "preprocess": info["config"]["preprocess"]["mode"],
                    "total": info["summary"]["total"],
                    "path": str(path.resolve()),
                }
            )
        except (OSError, ValueError, KeyError, TypeError):
            continue  # A corrupt/in-progress manifest must not hide other valid runs.
    return sorted(rows, key=lambda row: (row["started_at_utc"], row["run_id"]), reverse=True)[
        :limit
    ]


def load_run(manifest_path: Path) -> tuple[RunResult, dict]:
    path = manifest_path.resolve()
    info = json.loads(path.read_text(encoding="utf-8"))
    if info["status"] == "running":
        raise ValueError("Analisis masih berlangsung. Tunggu sampai selesai.")
    # Relative-to-run names also work when a whole output folder has been moved.
    artifacts = {"run_json": str(path)}
    for name, value in info.get("artifact_paths", {}).items():
        filename = Path(value).name
        candidate = (path.parent / filename).resolve()
        if candidate.parent == path.parent and candidate.is_file():
            artifacts[name] = str(candidate)
    if info["status"] == "completed":
        for key in ("events_csv", "counts_csv", "annotated_video"):
            if key not in artifacts:
                raise ValueError(f"Berkas hasil tidak lengkap: {key}.")
    return (
        RunResult(
            run_id=info["run_id"],
            status=info["status"],
            summary=CountSummary(**info["summary"]),
            output_dir=str(path.parent),
            artifact_paths=artifacts,
            warnings=info.get("warnings", []),
        ),
        info,
    )
