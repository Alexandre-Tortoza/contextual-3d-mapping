"""Dense visual feature extraction boundary.

Issue: #161.
"""

from __future__ import annotations

from typing import Protocol

from visual_perception.config import FeatureExtractionConfig
from visual_perception.domain.feature_map import FeatureMap
from visual_perception.domain.image_payload import ImagePayload


class DenseFeatureExtractor(Protocol):
    """Extracts a spatial feature grid from an image or crop."""

    def extract(self, image: ImagePayload, config: FeatureExtractionConfig) -> FeatureMap:
        """Return a dense, finite feature map aligned to ``image``."""
        ...
