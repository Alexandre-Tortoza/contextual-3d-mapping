"""Image-space geometry contracts and coordinate invariants.

Issue: #155.

Coordinate convention (binding for the whole module):

- the pixel origin ``(0, 0)`` is the top-left corner of the image;
- ``x`` increases to the right, ``y`` increases downward;
- a :class:`BoundingBox` is stored as ``(x_min, y_min, x_max, y_max)`` in a
  half-open interval: ``x_min``/``y_min`` are inclusive, ``x_max``/``y_max``
  are exclusive, matching ``array[y_min:y_max, x_min:x_max]`` slicing;
- a :class:`Mask` is a boolean array shaped ``(height, width)`` aligned to a
  specific image resolution recorded on the mask itself.

Every region emitted by region discovery, tiling, merge, or serialization
must satisfy these invariants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """A half-open, axis-aligned pixel box: ``[x_min, x_max) x [y_min, y_max)``."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError(
                "BoundingBox must have positive width and height: "
                f"({self.x_min}, {self.y_min}, {self.x_max}, {self.y_max})."
            )

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    def clipped_to(self, *, width: int, height: int) -> BoundingBox:
        """Clip the box to an image of the given resolution."""
        x_min = max(0.0, min(self.x_min, width))
        y_min = max(0.0, min(self.y_min, height))
        x_max = max(0.0, min(self.x_max, width))
        y_max = max(0.0, min(self.y_max, height))
        return BoundingBox(x_min, y_min, x_max, y_max)


@dataclass(frozen=True, eq=False)
class Mask:
    """A boolean occupancy mask aligned to a specific image resolution."""

    data: np.ndarray
    image_width: int
    image_height: int

    def __post_init__(self) -> None:
        if self.data.dtype != np.bool_:
            raise ValueError(f"Mask dtype must be bool, got {self.data.dtype}.")
        if self.data.shape != (self.image_height, self.image_width):
            raise ValueError(
                "Mask shape must be (image_height, image_width) = "
                f"({self.image_height}, {self.image_width}), got {self.data.shape}."
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mask):
            return NotImplemented
        return (
            self.image_width == other.image_width
            and self.image_height == other.image_height
            and bool(np.array_equal(self.data, other.data))
        )

    @property
    def is_empty(self) -> bool:
        return not bool(self.data.any())

    def bounding_box(self) -> BoundingBox:
        """Compute the tight bounding box of the occupied pixels."""
        if self.is_empty:
            raise ValueError("Cannot compute a bounding box for an empty mask.")
        rows = np.any(self.data, axis=1)
        cols = np.any(self.data, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return BoundingBox(float(x_min), float(y_min), float(x_max) + 1.0, float(y_max) + 1.0)

    def area(self) -> int:
        return int(self.data.sum())

    def iou(self, other: Mask) -> float:
        """Intersection-over-union with another mask of the same resolution."""
        self._require_compatible(other)
        intersection = np.logical_and(self.data, other.data).sum()
        union = np.logical_or(self.data, other.data).sum()
        return float(intersection) / float(union) if union > 0 else 0.0

    def containment_ratio(self, other: Mask) -> float:
        """Fraction of ``other`` covered by ``self`` (0 when ``other`` is empty)."""
        self._require_compatible(other)
        other_area = other.area()
        if other_area == 0:
            return 0.0
        intersection = np.logical_and(self.data, other.data).sum()
        return float(intersection) / float(other_area)

    def _require_compatible(self, other: Mask) -> None:
        if (self.image_width, self.image_height) != (other.image_width, other.image_height):
            raise ValueError("Masks must share the same image resolution to be compared.")


@dataclass(frozen=True)
class CoordinateTransform:
    """An affine mapping between one local frame (tile, crop, resize) and the
    original image, expressed as ``global = local * scale + offset``.
    """

    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float

    def __post_init__(self) -> None:
        if self.scale_x <= 0 or self.scale_y <= 0:
            raise ValueError("CoordinateTransform scale factors must be positive.")

    @staticmethod
    def identity() -> CoordinateTransform:
        return CoordinateTransform(1.0, 1.0, 0.0, 0.0)

    def box_to_global(self, box: BoundingBox) -> BoundingBox:
        return BoundingBox(
            x_min=box.x_min * self.scale_x + self.offset_x,
            y_min=box.y_min * self.scale_y + self.offset_y,
            x_max=box.x_max * self.scale_x + self.offset_x,
            y_max=box.y_max * self.scale_y + self.offset_y,
        )

    def box_to_local(self, box: BoundingBox) -> BoundingBox:
        return BoundingBox(
            x_min=(box.x_min - self.offset_x) / self.scale_x,
            y_min=(box.y_min - self.offset_y) / self.scale_y,
            x_max=(box.x_max - self.offset_x) / self.scale_x,
            y_max=(box.y_max - self.offset_y) / self.scale_y,
        )

    def mask_to_global(self, mask: Mask, *, global_width: int, global_height: int) -> Mask:
        """Place a local mask onto a full-size canvas in global coordinates.

        Uses nearest-neighbor placement, which is exact for integer tile
        offsets/scale=1 (tiling) and approximate for non-unit scale (resize).
        """
        canvas = np.zeros((global_height, global_width), dtype=np.bool_)
        local_ys, local_xs = np.where(mask.data)
        if local_ys.size == 0:
            return Mask(canvas, global_width, global_height)
        global_xs = np.clip(
            np.round(local_xs * self.scale_x + self.offset_x).astype(np.int64),
            0,
            global_width - 1,
        )
        global_ys = np.clip(
            np.round(local_ys * self.scale_y + self.offset_y).astype(np.int64),
            0,
            global_height - 1,
        )
        canvas[global_ys, global_xs] = True
        return Mask(canvas, global_width, global_height)
