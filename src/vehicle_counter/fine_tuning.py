"""Prepare reviewed labels and run explicit, reproducible local fine-tuning."""

import argparse
import json
import re
import shutil
from collections import Counter
from math import isfinite
from pathlib import Path

import cv2
import yaml

from vehicle_counter.contracts import VEHICLE_CLASSES
from vehicle_counter.exporting import file_sha256, write_json
from vehicle_counter.training_labels import validate_boxes, yolo_text


def prepare_dataset(manifest: Path, output: Path) -> Path:
    """Require visual review and disjoint source groups before writing YOLO data."""
    records = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest label harus berisi daftar frame yang sudah diperiksa.")
    groups: dict[str, str] = {}
    hashes: dict[str, str] = {}
    samples = []
    counts = {split: Counter() for split in ("train", "val", "test")}
    frames = Counter()
    for number, record in enumerate(records):
        split = record.get("split")
        if split not in counts or record.get("reviewed") is not True or not record.get("reviewer"):
            raise ValueError("Setiap frame memerlukan split, reviewed=true, dan nama pemeriksa.")
        group = record.get("source_group", "")
        if not re.fullmatch(r"[0-9a-f]{64}", group):
            raise ValueError("source_group harus checksum video asli, termasuk untuk crop/CFR.")
        if group in groups and groups[group] != split:
            raise ValueError("Video sumber yang sama tidak boleh masuk split berbeda.")
        groups[group] = split
        image = Path(record["image"])
        if not image.is_absolute():
            image = manifest.parent / image
        frame = cv2.imread(str(image))
        if frame is None:
            raise ValueError(f"Gambar label tidak dapat dibaca: {image}")
        digest = file_sha256(image)
        if digest in hashes:
            raise ValueError("Gambar identik terduplikasi pada dataset.")
        hashes[digest] = split
        height, width = frame.shape[:2]
        boxes = validate_boxes(
            [
                {
                    "class_name": b["class_name"],
                    "bbox": [
                        b["bbox"][0] / width,
                        b["bbox"][1] / height,
                        b["bbox"][2] / width,
                        b["bbox"][3] / height,
                    ],
                }
                for b in record["boxes"]
            ]
        )
        counts[split].update(b["class_name"] for b in boxes)
        frames[split] += 1
        samples.append((image, split, f"{group[:16]}_{number:05d}", boxes, digest, record))
    if any(not frames[split] for split in counts):
        raise ValueError("Split train, val, dan test harus memiliki gambar.")
    if output.exists():
        raise ValueError("Pilih folder dataset baru; dataset lama tidak ditimpa.")
    output.mkdir(parents=True)
    provenance = []
    for image, split, stem, boxes, digest, record in samples:
        for folder in (output / "images" / split, output / "labels" / split):
            folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image, output / "images" / split / f"{stem}{image.suffix}")
        (output / "labels" / split / f"{stem}.txt").write_text(yolo_text(boxes), encoding="utf-8")
        provenance.append({**record, "dataset_stem": stem, "image_sha256": digest})
    yaml_path = output / "dataset.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(output.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": dict(enumerate(VEHICLE_CLASSES)),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    write_json(
        output / "provenance.json",
        {
            "annotation_manifest_sha256": file_sha256(manifest),
            "source_groups": groups,
            "frames_per_split": dict(frames),
            "boxes_per_split": counts,
            "missing_train_classes": sorted(set(VEHICLE_CLASSES) - set(counts["train"])),
            "human_reviewed": all(r["reviewer"] == "human" for r in records),
            "records": provenance,
        },
    )
    return yaml_path


def remap_dataset(data: Path, names: dict[int, str], output: Path) -> Path:
    """Align labels with a retained pretrained head by class name, never raw index."""
    config = yaml.safe_load(data.read_text(encoding="utf-8"))
    source_names = config["names"]
    if isinstance(source_names, list):
        source_names = dict(enumerate(source_names))
    inverse = {name: index for index, name in names.items()}
    if any(name not in inverse for name in source_names.values()):
        raise ValueError("Kelas dataset tidak tersedia pada kepala pretrained.")
    mapping = {int(index): inverse[name] for index, name in source_names.items()}
    source = Path(config["path"])
    if not source.is_absolute():
        source = data.parent / source
    if output.exists():
        raise ValueError("Folder dataset remap sudah ada; gunakan project baru.")
    output.mkdir(parents=True)
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
        for image in (source / "images" / split).iterdir():
            shutil.copyfile(image, output / "images" / split / image.name)
            label = source / "labels" / split / f"{image.stem}.txt"
            lines = label.read_text(encoding="utf-8").splitlines()
            converted = [
                f"{mapping[int(line.split()[0])]} {' '.join(line.split()[1:])}" for line in lines
            ]
            (output / "labels" / split / label.name).write_text(
                "\n".join(converted) + ("\n" if converted else ""), encoding="utf-8"
            )
    result = output / "dataset.yaml"
    result.write_text(
        yaml.safe_dump(
            {
                "path": str(output.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": names,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    write_json(output / "class_mapping.json", {"source_to_model": mapping})
    return result


def train(
    data: Path,
    weights: Path,
    project: Path,
    *,
    epochs: int,
    device: str,
    keep_pretrained_head: bool = False,
    freeze: int = 10,
    learning_rate: float = 0.0002,
) -> Path:
    import torch
    from ultralytics import YOLO

    if not data.is_file() or not weights.is_file():
        raise ValueError("Dataset YAML dan bobot awal harus berupa berkas lokal.")
    if device != "cpu" and not torch.cuda.is_available():
        raise ValueError("GPU CUDA belum tersedia pada lingkungan training.")
    if epochs < 1:
        raise ValueError("epochs minimal 1.")
    if freeze < 0 or not isfinite(learning_rate) or not 0 < learning_rate < 1:
        raise ValueError("freeze harus >= 0 dan learning-rate harus pada rentang (0, 1).")
    model = YOLO(str(weights.resolve()))
    original_data = data
    if keep_pretrained_head:
        data = remap_dataset(data, model.names, project / "dataset-retained-head")
    model.train(
        data=str(data.resolve()),
        epochs=epochs,
        imgsz=640,
        batch=4,
        device=device,
        workers=0,
        seed=37,
        deterministic=True,
        optimizer="AdamW",
        lr0=learning_rate,
        lrf=0.1,
        freeze=freeze,
        weight_decay=0.0001,
        warmup_epochs=2,
        patience=10,
        mosaic=0.2,
        close_mosaic=5,
        mixup=0,
        degrees=0,
        translate=0.05,
        scale=0.15,
        fliplr=0.2,
        flipud=0,
        hsv_h=0.01,
        hsv_s=0.2,
        hsv_v=0.2,
        amp=False,
        cache=False,
        project=str(project.resolve()),
        name="yolo26n-local",
        exist_ok=False,
        plots=True,
        save=True,
        val=True,
    )
    best = Path(model.trainer.best)
    write_json(
        best.parent.parent / "training_provenance.json",
        {
            "initial_weights": str(weights.resolve()),
            "initial_sha256": file_sha256(weights),
            "dataset_yaml": str(data.resolve()),
            "original_dataset_yaml": str(original_data.resolve()),
            "retained_pretrained_head": keep_pretrained_head,
            "dataset_yaml_sha256": file_sha256(data),
            "best_weights": str(best.resolve()),
            "best_sha256": file_sha256(best),
            "torch_version": torch.__version__,
            "cuda": torch.version.cuda,
            "device": device,
            "epochs_requested": epochs,
            "epochs_completed": model.trainer.epoch + 1,
            "model_class_names": model.names,
            "test_used_for_training_or_selection": False,
        },
    )
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("manifest", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    fit = commands.add_parser("train")
    fit.add_argument("data", type=Path)
    fit.add_argument("--weights", type=Path, default=Path("models/yolo26n.pt"))
    fit.add_argument("--project", type=Path, default=Path("outputs/fine-tuning"))
    fit.add_argument("--epochs", type=int, default=30)
    fit.add_argument("--device", default="0")
    fit.add_argument("--keep-pretrained-head", action="store_true")
    fit.add_argument("--freeze", type=int, default=10)
    fit.add_argument("--learning-rate", type=float, default=0.0002)
    args = parser.parse_args()
    if args.command == "prepare":
        path = prepare_dataset(args.manifest, args.output)
    else:
        path = train(
            args.data,
            args.weights,
            args.project,
            epochs=args.epochs,
            device=args.device,
            keep_pretrained_head=args.keep_pretrained_head,
            freeze=args.freeze,
            learning_rate=args.learning_rate,
        )
    print(json.dumps({"output": str(path.resolve())}))


if __name__ == "__main__":
    main()
