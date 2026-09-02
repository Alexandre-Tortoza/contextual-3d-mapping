"""Resolved pixel payload used internally by stages, ports, and fakes.

``ImageObservation`` (see #153) references pixel data only through an
``ArtifactReference`` and stays implementation-agnostic. Once a stage needs to
actually run inference, it resolves that reference to an ``ImagePayload``
(a concrete pixel array). This keeps the public contract free of imaging
library types while giving ports something real to operate on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class ImagePayload:
    """A concrete RGB pixel array plus the resolution it declares."""

    pixels: np.ndarray
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.pixels.ndim != 3 or self.pixels.shape[2] != 3:
            raise ValueError(f"pixels must have shape (H, W, 3), got {self.pixels.shape}.")
        if self.pixels.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"pixels shape {self.pixels.shape[:2]} does not match "
                f"(height, width) = ({self.height}, {self.width})."
            )

    def crop(self, x_min: int, y_min: int, x_max: int, y_max: int) -> ImagePayload:
        """Return a local, tile-scoped payload for the given half-open pixel box."""
        sub = self.pixels[y_min:y_max, x_min:x_max]
        return ImagePayload(sub, width=sub.shape[1], height=sub.shape[0])
