"""Data contracts shared by the image-context pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImageSample:
    """One image extracted from a source dataset."""

    frame_id: str
    source_index: int
    timestamp_ns: int
    width: int
    height: int
    image_path: Path


@dataclass(frozen=True)
class BoundingBox:
    """Pixel-aligned bounding box in xyxy format."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("Bounding box must have positive area.")


JsonObject = dict[str, Any]
