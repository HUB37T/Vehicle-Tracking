"""Load strict YAML configuration without model or UI dependencies."""

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import yaml

from vehicle_counter.contracts import (
    CongestionConfig,
    CountingLineConfig,
    PreprocessConfig,
    RunConfig,
    TrafficZoneConfig,
)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} harus berupa mapping dengan nama field berupa teks.")
    return dict(value)


def _known_keys(data: dict[str, Any], cls: type, name: str) -> None:
    unknown = set(data) - {item.name for item in fields(cls)}
    if unknown:
        raise ValueError(f"Field {name} tidak dikenal: {', '.join(sorted(unknown))}.")


def load_config(path: str | Path) -> RunConfig:
    """Read a YAML file; reject unknown keys and invalid values with ValueError."""
    try:
        return config_from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    except yaml.YAMLError as error:
        raise ValueError(f"Format konfigurasi tidak valid: {error}") from error


def config_from_dict(value: Any) -> RunConfig:
    try:
        data = _mapping(value, "config")
        _known_keys(data, RunConfig, "config")
        for key, cls in (
            ("preprocess", PreprocessConfig),
            ("counting_line", CountingLineConfig),
            ("congestion", CongestionConfig),
        ):
            if key in data:
                nested = _mapping(data[key], key)
                _known_keys(nested, cls, key)
                data[key] = cls(**nested)
        if "traffic_zones" in data:
            if not isinstance(data["traffic_zones"], (list, tuple)):
                raise ValueError("traffic_zones harus berupa daftar.")
            zones = []
            for value in data["traffic_zones"]:
                nested = _mapping(value, "traffic_zones")
                _known_keys(nested, TrafficZoneConfig, "traffic_zones")
                zones.append(TrafficZoneConfig(**nested))
            data["traffic_zones"] = tuple(zones)
        return RunConfig(**data)
    except (TypeError, yaml.YAMLError) as error:
        raise ValueError(f"Format konfigurasi tidak valid: {error}") from error


def config_to_dict(config: RunConfig) -> dict[str, Any]:
    """Return a detached, JSON-serializable configuration snapshot."""
    return asdict(config)
