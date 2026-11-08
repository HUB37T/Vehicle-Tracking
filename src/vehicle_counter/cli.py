"""Local model preparation, video inspection, and full analysis commands."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from vehicle_counter.config import load_config
from vehicle_counter.contracts import PREPROCESS_MODES, RunConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Penghitung kendaraan dari video MP4 lokal.")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-model", help="Unduh bobot pretrained YOLO11n resmi.")
    prepare.add_argument("--directory", type=Path, default=Path("models"))
    prepare.add_argument("--source", choices=["huggingface", "github"], default="huggingface")
    inspect = commands.add_parser("inspect", help="Periksa metadata dan frame awal video.")
    inspect.add_argument("source", type=Path)
    evaluate = commands.add_parser("evaluate", help="Bandingkan run dengan anotasi manual.")
    evaluate.add_argument("ground_truth", type=Path, help="CSV dengan sidecar JSON bernama sama.")
    evaluate.add_argument("runs", type=Path, nargs="+", help="Folder hasil run yang sudah selesai.")
    evaluate.add_argument("--clip-id", required=True)
    evaluate.add_argument("--tolerance", type=float, default=0.5)
    evaluate.add_argument("--output", type=Path)
    lab = commands.add_parser("pcd-lab", help="Ekspor tahapan pengolahan citra dari dua frame.")
    lab.add_argument("source", type=Path)
    lab.add_argument("--frame", type=int, default=30, help="Indeks frame target, mulai dari 1.")
    lab.add_argument("--min-area", type=int, default=100, help="Luas minimum komponen (piksel).")
    lab.add_argument("--config", type=Path, help="Konfigurasi parameter prapemrosesan.")
    lab.add_argument("--output-root", type=Path, default=Path("outputs"))
    analyze = commands.add_parser("analyze", help="Jalankan deteksi, tracking, dan counting.")
    analyze.add_argument("source", type=Path)
    analyze.add_argument("--config", type=Path, help="Default tanpa berkas: RunConfig().")
    analyze.add_argument("--output-root", type=Path, default=Path("outputs"))
    analyze.add_argument("--device", help="Ganti device pada konfigurasi, misalnya cpu atau 0.")
    analyze.add_argument("--weights", help="Ganti path bobot lokal pada konfigurasi.")
    analyze.add_argument("--tracker-config", help="Ganti nama atau path YAML tracker.")
    analyze.add_argument("--preprocess", choices=PREPROCESS_MODES, help="Ganti mode prapemrosesan.")
    args = parser.parse_args()
    try:
        if args.command == "prepare-model":
            from vehicle_counter.exporting import file_sha256
            from vehicle_counter.tracking import prepare_model

            path = prepare_model(args.directory, source=args.source)
            result = {"weights_path": str(path), "sha256": file_sha256(path)}
        elif args.command == "inspect":
            from vehicle_counter.video import inspect_video

            result = asdict(inspect_video(args.source))
        elif args.command == "evaluate":
            from vehicle_counter.evaluation import compare_runs
            from vehicle_counter.exporting import write_json

            result = compare_runs(args.ground_truth, args.clip_id, args.runs, args.tolerance)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                write_json(args.output, result)
        elif args.command == "pcd-lab":
            from vehicle_counter.pcd_lab import export_lab

            config = load_config(args.config) if args.config else RunConfig()
            result = export_lab(
                args.source,
                args.output_root,
                config.preprocess,
                frame_index=args.frame,
                min_area=args.min_area,
            )
        else:
            from vehicle_counter.pipeline import process_video

            config = load_config(args.config) if args.config else RunConfig()
            if args.device:
                config = replace(config, device=args.device)
            if args.weights:
                config = replace(config, model_weights=args.weights)
            if args.tracker_config:
                config = replace(config, tracker_config=args.tracker_config)
            if args.preprocess:
                config = replace(
                    config, preprocess=replace(config.preprocess, mode=args.preprocess)
                )
            run = process_video(args.source, config, args.output_root)
            result = asdict(run)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            if run.status != "completed":
                parser.exit(1, f"Analisis gagal. Periksa {Path(run.output_dir) / 'run.json'}\n")
            return
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Tidak dapat menjalankan perintah: {error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
