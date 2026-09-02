"""Real language-aligned embedding backend adapter.

Issue: #188. Blocked on #174 and a GPU-equipped environment.
"""

from __future__ import annotations

from visual_perception.config import LanguageEmbeddingConfig
from visual_perception.domain.errors import BackendUnavailableError
from visual_perception.domain.image_payload import ImagePayload


class RealLanguageAlignedEncoderAdapter:
    """Satisfies :class:`~visual_perception.ports.language_embedding.LanguageAlignedEncoder`.

    Not implemented yet. Use
    ``infrastructure.fakes.fake_language_encoder.FakeLanguageAlignedEncoder``
    for tests and development.
    """

    def encode_image(self, image: ImagePayload, config: LanguageEmbeddingConfig) -> tuple[float, ...]:
        raise BackendUnavailableError(
            f"Language-aligned encoder backend {config.backend!r} is not implemented in this "
            "environment (no GPU, no benchmark-selected checkpoint). See issue #188."
        )

    def encode_text(self, text: str, config: LanguageEmbeddingConfig) -> tuple[float, ...]:
        raise BackendUnavailableError(
            f"Language-aligned encoder backend {config.backend!r} is not implemented in this "
            "environment (no GPU, no benchmark-selected checkpoint). See issue #188."
        )
