"""High-resolution mask-aware region pooling.

Issue: #162.

Two pooling methods are supported behind one function:

- ``patch_grid_baseline``: includes a feature-grid cell only when its pixel
  *center* falls inside the region mask, then averages unweighted. Simple,
  but a mask smaller than one grid cell has no cell center inside it and is
  rejected.
- ``pixel_nearest_highres``: gathers, for every mask pixel, the feature
  vector of its nearest grid cell, then averages. This keeps small regions
  representable whenever at least one mask pixel has feature support.

Both paths L2-normalize the resulting vector (documented normalization
behavior); a normalization is skipped only when unreachable (finite, nonzero
input), which cannot happen once a method has accepted a mask.
"""

from __future__ import annotations

import numpy as np

from visual_perception.domain.embeddings import VisualEmbedding
from visual_perception.domain.feature_map import FeatureMap
from visual_perception.domain.geometry import Mask
from visual_perception.domain.regions import ObservedRegion

BASELINE = "patch_grid_baseline"
HIGH_RESOLUTION = "pixel_nearest_highres"
_METHODS = frozenset({BASELINE, HIGH_RESOLUTION})


def pool_region_vector(mask: Mask, feature_map: FeatureMap, method: str) -> tuple[float, ...]:
    """Pool one region's dense features into a single finite, L2-normalized vector."""
    if method not in _METHODS:
        raise ValueError(f"Unknown pooling method {method!r}, expected one of {sorted(_METHODS)}.")
    stride_x = mask.image_width / feature_map.grid_width
    stride_y = mask.image_height / feature_map.grid_height
    if method == BASELINE:
        vector = _pool_patch_grid_baseline(mask, feature_map, stride_x, stride_y)
    else:
        vector = _pool_pixel_nearest(mask, feature_map, stride_x, stride_y)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Pooled vector has zero norm and cannot be normalized.")
    return tuple((vector / norm).tolist())


def _pool_patch_grid_baseline(
    mask: Mask, feature_map: FeatureMap, stride_x: float, stride_y: float
) -> np.ndarray:
    included = []
    for grid_y in range(feature_map.grid_height):
        center_y = int((grid_y + 0.5) * stride_y)
        if not 0 <= center_y < mask.image_height:
            continue
        for grid_x in range(feature_map.grid_width):
            center_x = int((grid_x + 0.5) * stride_x)
            if 0 <= center_x < mask.image_width and mask.data[center_y, center_x]:
                included.append(feature_map.data[grid_y, grid_x])
    if not included:
        raise ValueError(
            "No feature-grid cell center falls inside the mask; the region cannot be aligned "
            f"with the '{BASELINE}' method. Try '{HIGH_RESOLUTION}' for small regions."
        )
    return np.mean(np.stack(included), axis=0)


def _pool_pixel_nearest(
    mask: Mask, feature_map: FeatureMap, stride_x: float, stride_y: float
) -> np.ndarray:
    ys, xs = np.where(mask.data)
    if ys.size == 0:
        raise ValueError("Cannot pool an empty mask.")
    grid_ys = np.clip((ys / stride_y).astype(np.int64), 0, feature_map.grid_height - 1)
    grid_xs = np.clip((xs / stride_x).astype(np.int64), 0, feature_map.grid_width - 1)
    gathered = feature_map.data[grid_ys, grid_xs]
    return np.mean(gathered, axis=0)


def pool_regions(
    regions: tuple[ObservedRegion, ...],
    feature_map: FeatureMap,
    method: str = HIGH_RESOLUTION,
) -> tuple[VisualEmbedding, ...]:
    """Pool every region against one dense feature map into a :class:`VisualEmbedding`."""
    embeddings = []
    for region in regions:
        vector = pool_region_vector(region.mask, feature_map, method)
        embeddings.append(
            VisualEmbedding(
                embedding_id=f"visual-{region.region_id}",
                region_id=region.region_id,
                vector=vector,
                dimension=len(vector),
                pooling_method=method,
                feature_resolution=f"{feature_map.grid_width}x{feature_map.grid_height}",
                model_id=feature_map.model_id,
                normalized=True,
            )
        )
    return tuple(embeddings)
