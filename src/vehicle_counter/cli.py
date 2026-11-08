"""Local model preparation, video inspection, and full analysis commands."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from vehicle_counter.config import load_config
from vehicle_counter.contracts import RunConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Penghitung kendaraan dari video MP4 lokal.")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-model", help="Unduh bobot pretrained YOLO11n resmi.")
    prepare.add_argument("--directory", type=Path, default=Path("models"))
    prepare.add_argument("--source", choices=["huggingface", "github"], default="huggingface")
    inspect = commands.add_parser("inspect", help="Periksa metadata dan frame awal video.")
    inspect.add_argument("source", type=Path)
    analyze = commands.add_parser("analyze", help="Jalankan deteksi, tracking, dan counting.")
    analyze.add_argument("source", type=Path)
    analyze.add_argument("--config", type=Path, help="Default tanpa berkas: RunConfig().")
    analyze.add_argument("--output-root", type=Path, default=Path("outputs"))
    analyze.add_argument("--device", help="Ganti device pada konfigurasi, misalnya cpu atau 0.")
    analyze.add_argument("--weights", help="Ganti path bobot lokal pada konfigurasi.")
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
        else:
            from vehicle_counter.pipeline import process_video

            config = load_config(args.config) if args.config else RunConfig()
            if args.device:
                config = replace(config, device=args.device)
            if args.weights:
                config = replace(config, model_weights=args.weights)
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
