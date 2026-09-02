"""Dense visual feature map contract.

Issue: #161.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class FeatureMap:
    """A spatial grid of dense visual features aligned to one image.

    ``stride_x``/``stride_y`` describe how many original-image pixels one
    feature-grid cell covers, which is exactly the metadata mask-aware
    pooling (#162) needs to align a region mask with this grid.
    """

    data: np.ndarray  # shape (grid_height, grid_width, dimension)
    stride_x: float
    stride_y: float
    dimension: int
    model_id: str

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError(f"FeatureMap.data must have shape (H, W, D), got {self.data.shape}.")
        if self.data.shape[2] != self.dimension:
            raise ValueError(
                f"FeatureMap dimension mismatch: declared {self.dimension}, "
                f"got last axis {self.data.shape[2]}."
            )
        if self.stride_x <= 0 or self.stride_y <= 0:
            raise ValueError("FeatureMap strides must be positive.")
        if not np.isfinite(self.data).all():
            raise ValueError("FeatureMap.data must be finite (no NaN/Inf).")

    @property
    def grid_height(self) -> int:
        return int(self.data.shape[0])

    @property
    def grid_width(self) -> int:
        return int(self.data.shape[1])
