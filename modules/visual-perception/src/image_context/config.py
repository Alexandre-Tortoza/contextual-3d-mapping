"""Typed YAML configuration for the executable pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    """ROS bag image source settings."""

    bag_path: Path
    image_topic: str
    sample_size: int = 10
    seed: int = 42


@dataclass(frozen=True)
class PipelineConfig:
    """Complete run configuration."""

    dataset: DatasetConfig
    output_directory: Path = Path("runs")
    run_id: str = "corridor02-sample"


def load_config(path: Path) -> PipelineConfig:
    """Load configuration and resolve filesystem paths relative to the YAML file."""
    with path.open(encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be an object.")
    payload = cast(dict[str, Any], raw)
    dataset = _mapping(payload, "dataset")
    base = path.resolve().parent
    bag_path = _resolved_path(base, _string(dataset, "bag_path"))
    output_directory = _resolved_path(base, str(payload.get("output_directory", "runs")))
    config = PipelineConfig(
        dataset=DatasetConfig(
            bag_path=bag_path,
            image_topic=_string(dataset, "image_topic"),
            sample_size=int(dataset.get("sample_size", 10)),
            seed=int(dataset.get("seed", 42)),
        ),
        output_directory=output_directory,
        run_id=str(payload.get("run_id", "corridor02-sample")),
    )
    _validate(config)
    return config


def _validate(config: PipelineConfig) -> None:
    if config.dataset.sample_size <= 0:
        raise ValueError("dataset.sample_size must be positive.")
    if not config.dataset.image_topic:
        raise ValueError("dataset.image_topic must not be empty.")


def _mapping(payload: dict[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration field '{name}' must be an object.")
    return cast(dict[str, Any], value)


def _string(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Configuration field '{name}' must be a non-empty string.")
    return value


def _resolved_path(base: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else (base / path).resolve()
