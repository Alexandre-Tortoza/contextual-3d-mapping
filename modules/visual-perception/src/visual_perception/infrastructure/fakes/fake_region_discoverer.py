"""Deterministic, GPU-free fake for the region discovery boundary (#158)."""

from __future__ import annotations

import numpy as np

from visual_perception.config import RegionDiscoveryConfig
from visual_perception.domain.geometry import Mask
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.regions import LocalRegionProposal


class FakeRegionDiscoverer:
    """Finds connected components of non-background pixels.

    Deterministic and dependency-free: a caller controls region count by
    controlling the synthetic image content (a plain image yields zero
    regions, one colored blob yields one, several yield several).
    """

    def __init__(self, background_threshold: int = 250, min_area: int = 4) -> None:
        self.background_threshold = background_threshold
        self.min_area = min_area

    def discover(
        self, image: ImagePayload, config: RegionDiscoveryConfig
    ) -> tuple[LocalRegionProposal, ...]:
        foreground = image.pixels.min(axis=2) < self.background_threshold
        components = _connected_components(foreground)
        proposals = []
        for index, component in enumerate(components):
            if int(component.sum()) < self.min_area:
                continue
            mask = Mask(component, image.width, image.height)
            proposals.append(
                LocalRegionProposal(
                    local_id=f"fake-{index}",
                    mask=mask,
                    box=mask.bounding_box(),
                    geometric_confidence=max(config.score_threshold, 0.6),
                    source="fake_region_discoverer",
                )
            )
        return tuple(proposals)


def _connected_components(foreground: np.ndarray) -> list[np.ndarray]:
    """4-connectivity connected-component labeling with no external dependency."""
    visited = np.zeros_like(foreground, dtype=np.bool_)
    height, width = foreground.shape
    components: list[np.ndarray] = []
    for start_y in range(height):
        for start_x in range(width):
            if not foreground[start_y, start_x] or visited[start_y, start_x]:
                continue
            component = np.zeros_like(foreground, dtype=np.bool_)
            stack = [(start_y, start_x)]
            visited[start_y, start_x] = True
            while stack:
                y, x = stack.pop()
                component[y, x] = True
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and foreground[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            components.append(component)
    components.sort(key=lambda component: int(component.sum()), reverse=True)
    return components
