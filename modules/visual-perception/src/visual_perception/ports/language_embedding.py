"""Language-aligned region embedding boundary.

Issue: #163.

Kept separate from ``DenseFeatureExtractor`` (#161): a language-aligned
encoder produces a second, independent region representation, aligned with
text, that may live in a different space with a different lifecycle.
"""

from __future__ import annotations

from typing import Protocol

from visual_perception.config import LanguageEmbeddingConfig
from visual_perception.domain.image_payload import ImagePayload


class LanguageAlignedEncoder(Protocol):
    """Encodes region image evidence and text into a shared documented space."""

    def encode_image(
        self, image: ImagePayload, config: LanguageEmbeddingConfig
    ) -> tuple[float, ...]:
        """Return a finite embedding vector for one region crop."""
        ...

    def encode_text(self, text: str, config: LanguageEmbeddingConfig) -> tuple[float, ...]:
        """Return a finite embedding vector for a text query, when supported."""
        ...
