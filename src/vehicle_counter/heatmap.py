"""Bounded, time-weighted anchor presence map; separate from vehicle counts."""

from pathlib import Path

import cv2
import numpy as np

from vehicle_counter.exporting import write_json


class PresenceHeatmap:
    def __init__(self, width: int, height: int, fps: float):
        self.width, self.height, self.fps = width, height, fps
        self.seconds = np.zeros((128, 128), np.float64)
        self.frames = 0

    def update(self, observations):
        cells = set()
        for item in observations:
            x, y = item.anchor
            if 0 <= x < self.width and 0 <= y < self.height:
                cells.add(
                    (min(127, int(y / self.height * 128)), min(127, int(x / self.width * 128)))
                )
        for y, x in cells:
            self.seconds[y, x] += 1 / self.fps
        self.frames += 1

    def export(self, output: Path, source_frame) -> dict[str, str]:
        scale = min(1, 1280 / max(source_frame.shape[:2]))
        base = cv2.resize(source_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        smoothed = cv2.GaussianBlur(self.seconds.astype(np.float32), (5, 5), 0)
        peak = float(smoothed.max())
        levels = (
            np.rint(255 * smoothed / peak).astype(np.uint8)
            if peak
            else np.zeros((128, 128), np.uint8)
        )
        heat = cv2.applyColorMap(
            cv2.resize(levels, (base.shape[1], base.shape[0])), cv2.COLORMAP_TURBO
        )
        active = cv2.resize((smoothed > 0).astype(np.uint8), (base.shape[1], base.shape[0])) > 0
        overlay = base.copy()
        overlay[active] = cv2.addWeighted(base, 0.4, heat, 0.6, 0)[active]
        ok, encoded = cv2.imencode(".png", overlay)
        if not ok:
            raise RuntimeError("Gagal menulis heatmap.")
        path = output / "heatmap.png"
        path.write_bytes(encoded.tobytes())
        raw = output / "heatmap.npz"
        np.savez_compressed(raw, seconds=self.seconds)
        meta = output / "heatmap.json"
        write_json(
            meta,
            {
                "method": "binary_anchor_cell_presence_per_frame",
                "unit": "observed_seconds_per_cell",
                "grid": [128, 128],
                "frames": self.frames,
                "fps": self.fps,
                "maximum_raw_seconds": float(self.seconds.max()),
                "display_gaussian_kernel": 5,
                "maximum_smoothed_seconds": peak,
                "note": (
                    "Warna relatif dalam run ini. Bukan jumlah kendaraan unik atau bukti macet."
                ),
            },
        )
        return {"heatmap_png": str(path), "heatmap_npz": str(raw), "heatmap_json": str(meta)}
