"""Portable camera geometry and the local preset library."""

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from vehicle_counter.contracts import BBox, CongestionConfig, CountingLineConfig, Point

PRESET_FORMAT = "vehicle-tracking-camera-preset"
PRESET_VERSION = 1
MAX_NAME_LENGTH = 80
MAX_PRESET_BYTES = 16 * 1024


@dataclass(frozen=True, slots=True)
class CameraPreset:
    name: str
    p1: Point
    p2: Point
    roi: BBox

    def __post_init__(self) -> None:
        name = self.name.strip() if isinstance(self.name, str) else ""
        if not name or len(name) > MAX_NAME_LENGTH or not name.isprintable():
            raise ValueError(f"Nama preset harus berupa teks 1–{MAX_NAME_LENGTH} karakter.")
        object.__setattr__(self, "name", name)
        for field in ("p1", "p2", "roi"):
            coordinates = getattr(self, field)
            if not isinstance(coordinates, (tuple, list)) or any(
                type(value) not in (float, int) or not 0 <= value <= 1 for value in coordinates
            ):
                raise ValueError(f"Koordinat {field} harus berupa angka antara 0 dan 1.")
            object.__setattr__(self, field, tuple(map(float, coordinates)))
        CountingLineConfig(p1=self.p1, p2=self.p2)
        CongestionConfig(roi=self.roi)

    @property
    def key(self) -> str:
        return self.name.casefold()

    def points(self) -> dict[str, list[float]]:
        return {"p1": list(self.p1), "p2": list(self.p2), "roi": list(self.roi)}

    def to_json(self) -> str:
        payload = {
            "format": PRESET_FORMAT,
            "version": PRESET_VERSION,
            "name": self.name,
            **self.points(),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, data: bytes) -> "CameraPreset":
        if len(data) > MAX_PRESET_BYTES:
            raise ValueError("File preset maksimal 16 KB.")
        try:
            payload = json.loads(data.decode("utf-8-sig"))
        except (ValueError, RecursionError) as error:
            raise ValueError("File preset harus berupa JSON yang valid.") from error
        expected_fields = {"format", "version", "name", "p1", "p2", "roi"}
        if not isinstance(payload, dict) or set(payload) != expected_fields:
            raise ValueError("Isi file preset kamera tidak lengkap atau memiliki kolom tambahan.")
        if payload["format"] != PRESET_FORMAT:
            raise ValueError("Format file bukan preset kamera Vehicle Tracking.")
        if type(payload["version"]) is not int or payload["version"] != PRESET_VERSION:
            raise ValueError("Versi file preset kamera belum didukung.")
        return cls(payload["name"], payload["p1"], payload["p2"], payload["roi"])


def load_camera_presets(path: Path) -> dict[str, CameraPreset]:
    if not path.is_file():
        return {}
    with closing(sqlite3.connect(path)) as connection:
        if not connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'camera_presets'"
        ).fetchone():
            return {}
        return {
            key: CameraPreset.from_json(payload.encode("utf-8"))
            for key, payload in connection.execute("SELECT name_key, payload FROM camera_presets")
        }


def save_camera_preset(path: Path, preset: CameraPreset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = preset.to_json()
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS camera_presets (name_key TEXT PRIMARY KEY, payload TEXT)"
        )
        cursor = connection.execute(
            "INSERT INTO camera_presets (name_key, payload) VALUES (?, ?) "
            "ON CONFLICT (name_key) DO NOTHING",
            (preset.key, payload),
        )
        if cursor.rowcount == 0:
            existing = connection.execute(
                "SELECT payload FROM camera_presets WHERE name_key = ?", (preset.key,)
            ).fetchone()
            if existing != (payload,):
                raise ValueError("Nama preset sudah dipakai. Gunakan nama lain.")
