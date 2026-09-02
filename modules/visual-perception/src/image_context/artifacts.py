"""Durable JSON artifacts for one run."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from image_context.models import ImageSample
from image_context.serialization import sample_to_dict


class ArtifactRepository:
    """Owns the stable on-disk contract and resumable stage records."""

    def __init__(self, run_directory: Path) -> None:
        self.run_directory = run_directory

    def initialize(self, manifest: dict[str, Any], *, overwrite: bool) -> None:
        """Initialize a run or validate that an existing run is compatible."""
        manifest_path = self.run_directory / "manifest.json"
        if self.run_directory.exists() and overwrite:
            shutil.rmtree(self.run_directory)
        if manifest_path.exists():
            existing = self._read_json(manifest_path)
            if existing.get("fingerprint") != manifest.get("fingerprint"):
                raise ValueError(
                    f"Run directory '{self.run_directory}' contains a different configuration. "
                    "Use --overwrite or another --run-id."
                )
            return
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self._write_json(manifest_path, manifest)

    @property
    def frames_directory(self) -> Path:
        """Directory where extracted frame folders are stored."""
        return self.run_directory / "frames"

    def frame_directory(self, sample: ImageSample) -> Path:
        """Return one sample's artifact directory."""
        return self.frames_directory / sample.frame_id

    def write_selection(self, samples: tuple[ImageSample, ...]) -> None:
        """Persist selected source indices and extracted image metadata."""
        self._write_json(
            self.run_directory / "selected_frames.json",
            {"frames": [sample_to_dict(sample) for sample in samples]},
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open(encoding="utf-8") as json_file:
            result: dict[str, Any] = json.load(json_file)
        return result

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as json_file:
            json.dump(payload, json_file, indent=2, sort_keys=True)
            json_file.flush()
            os.fsync(json_file.fileno())
        os.replace(temporary, path)
