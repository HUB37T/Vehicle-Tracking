"""Explicitly synthetic observations for exercising the counter without YOLO/video."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from vehicle_counter.config import config_to_dict, load_config
from vehicle_counter.contracts import RunConfig, TrackObservation
from vehicle_counter.counting import VehicleCounter


def synthetic_frames() -> list[list[TrackObservation]]:
    """Five frames on a 101x101 canvas, designed for the default line at y=60.

    ID 1 crosses downward, ID 2 crosses upward, ID 3 stays above the line,
    ID 4 crosses its extension outside the finite segment, ID 5 stays neutral.
    These are fabricated track observations, not detector/tracker output.
    """
    frames = []
    paths = [
        (1, "car", 40.0, [25, 35, 45, 60, 75]),
        (2, "motorcycle", 70.0, [90, 80, 70, 60, 45]),
        (3, "bus", 50.0, [25, 25, 25, 25, 25]),
        (4, "truck", 98.0, [25, 35, 45, 60, 75]),
        (5, "car", 60.0, [60, 60, 60, 60, 60]),
    ]
    for frame_index in range(5):
        observations = []
        for track_id, class_name, x, ys in paths:
            y = float(ys[frame_index])
            observations.append(
                TrackObservation(
                    frame_index=frame_index,
                    timestamp_seconds=frame_index / 10,
                    track_id=track_id,
                    class_name=class_name,
                    confidence=0.9,
                    bbox_xyxy=(x - 3, y - 8, x + 3, y),
                )
            )
        frames.append(observations)
    return frames


def run_simulation(config: RunConfig | None = None) -> dict:
    config = config or RunConfig()
    counter = VehicleCounter("synthetic-demo", config, width=101, height=101)
    frames = synthetic_frames()
    for frame in frames:
        counter.update(frame)
    return {
        "source_kind": "synthetic",
        "is_model_inference": False,
        "notice": "Data sintetis untuk menguji counting; bukan evaluasi video atau YOLO.",
        "canvas": {"width": 101, "height": 101, "source_fps": 10},
        "processed_frames": len(frames),
        "config": config_to_dict(config),
        "events": [asdict(event) for event in counter.events],
        "summary": asdict(counter.summary()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulasi counting dengan data sintetis, tanpa YOLO."
    )
    parser.add_argument(
        "--config", type=Path, help="Konfigurasi YAML; default memakai RunConfig()."
    )
    parser.add_argument("--output", type=Path, help="Opsional: simpan JSON simulasi ke lokasi ini.")
    args = parser.parse_args()
    try:
        config = load_config(args.config) if args.config else RunConfig()
    except (OSError, ValueError) as error:
        parser.exit(2, f"Konfigurasi gagal dibaca: {error}\n")
    result = json.dumps(run_simulation(config), indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
