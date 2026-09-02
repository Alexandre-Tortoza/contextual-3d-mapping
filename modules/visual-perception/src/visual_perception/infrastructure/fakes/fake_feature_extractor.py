"""Deterministic, GPU-free fake for the dense feature extraction boundary (#161)."""

from __future__ import annotations

import numpy as np

from visual_perception.config import FeatureExtractionConfig
from visual_perception.domain.feature_map import FeatureMap
from visual_perception.domain.image_payload import ImagePayload

_DIMENSION = 4  # mean R, mean G, mean B, constant bias term.


class FakeDenseFeatureExtractor:
    """Average-pools raw pixels into a small grid: real signal, zero ML dependencies."""

    def extract(self, image: ImagePayload, config: FeatureExtractionConfig) -> FeatureMap:
        grid_h = max(1, min(config.feature_resolution, image.height))
        grid_w = max(1, min(config.feature_resolution, image.width))
        stride_y = image.height / grid_h
        stride_x = image.width / grid_w

        data = np.empty((grid_h, grid_w, _DIMENSION), dtype=np.float64)
        for gy in range(grid_h):
            y0, y1 = int(gy * stride_y), max(int(gy * stride_y) + 1, int((gy + 1) * stride_y))
            for gx in range(grid_w):
                x0, x1 = int(gx * stride_x), max(int(gx * stride_x) + 1, int((gx + 1) * stride_x))
                cell = image.pixels[y0:y1, x0:x1].astype(np.float64) / 255.0
                mean_rgb = cell.reshape(-1, 3).mean(axis=0)
                data[gy, gx] = (*mean_rgb, 1.0)

        return FeatureMap(
            data=data, stride_x=stride_x, stride_y=stride_y, dimension=_DIMENSION, model_id="fake-features"
        )
