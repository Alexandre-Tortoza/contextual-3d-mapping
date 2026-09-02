"""Deterministic, GPU-free fake for the language-aligned embedding boundary (#163)."""

from __future__ import annotations

import hashlib

import numpy as np

from visual_perception.config import LanguageEmbeddingConfig
from visual_perception.domain.image_payload import ImagePayload


class FakeLanguageAlignedEncoder:
    """Maps image/text content into a fixed-dimension space via a seeded hash.

    Not semantically meaningful, but deterministic, finite, and dimension-
    correct, which is all the boundary contract (#163) requires from a fake.
    """

    def encode_image(self, image: ImagePayload, config: LanguageEmbeddingConfig) -> tuple[float, ...]:
        mean_rgb = image.pixels.astype(np.float64).mean(axis=(0, 1)) / 255.0
        seed = int(hashlib.sha256(mean_rgb.tobytes()).hexdigest(), 16) % (2**32)
        return self._vector(seed, config.dimension)

    def encode_text(self, text: str, config: LanguageEmbeddingConfig) -> tuple[float, ...]:
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
        return self._vector(seed, config.dimension)

    @staticmethod
    def _vector(seed: int, dimension: int) -> tuple[float, ...]:
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=dimension)
        vector /= np.linalg.norm(vector)
        return tuple(vector.tolist())
