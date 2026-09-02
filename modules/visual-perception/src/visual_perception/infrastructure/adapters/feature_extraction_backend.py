"""Real dense visual feature backend adapter.

Issue: #187. Blocked on #174 and a GPU-equipped environment.
"""

from __future__ import annotations

from visual_perception.config import FeatureExtractionConfig
from visual_perception.domain.errors import BackendUnavailableError
from visual_perception.domain.feature_map import FeatureMap
from visual_perception.domain.image_payload import ImagePayload


class RealDenseFeatureExtractionAdapter:
    """Satisfies :class:`~visual_perception.ports.feature_extraction.DenseFeatureExtractor`.

    Not implemented yet. Use
    ``infrastructure.fakes.fake_feature_extractor.FakeDenseFeatureExtractor``
    for tests and development.
    """

    def extract(self, image: ImagePayload, config: FeatureExtractionConfig) -> FeatureMap:
        raise BackendUnavailableError(
            f"Dense feature backend {config.backend!r} is not implemented in this "
            "environment (no GPU, no benchmark-selected checkpoint). See issue #187."
        )
