"""Load strict YAML configuration without model or UI dependencies."""

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import yaml

from vehicle_counter.contracts import CountingLineConfig, PreprocessConfig, RunConfig


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
        data = _mapping(yaml.safe_load(Path(path).read_text(encoding="utf-8")), "config")
        _known_keys(data, RunConfig, "config")
        for key, cls in (("preprocess", PreprocessConfig), ("counting_line", CountingLineConfig)):
            if key in data:
                nested = _mapping(data[key], key)
                _known_keys(nested, cls, key)
                data[key] = cls(**nested)
        return RunConfig(**data)
    except (TypeError, yaml.YAMLError) as error:
        raise ValueError(f"Format konfigurasi tidak valid: {error}") from error


def config_to_dict(config: RunConfig) -> dict[str, Any]:
    """Return a detached, JSON-serializable configuration snapshot."""
    return asdict(config)
