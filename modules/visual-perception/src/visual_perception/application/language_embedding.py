"""Language-aligned region embedding stage.

Issue: #163.
"""

from __future__ import annotations

from visual_perception.config import LanguageEmbeddingConfig
from visual_perception.domain.embeddings import LanguageEmbedding
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.regions import ObservedRegion
from visual_perception.ports.language_embedding import LanguageAlignedEncoder


def encode_regions(
    regions: tuple[ObservedRegion, ...],
    image: ImagePayload,
    encoder: LanguageAlignedEncoder,
    config: LanguageEmbeddingConfig,
) -> tuple[LanguageEmbedding, ...]:
    """Encode every region's crop into the language-aligned embedding space."""
    embeddings = []
    for region in regions:
        box = region.box
        crop = image.crop(int(box.x_min), int(box.y_min), int(box.x_max), int(box.y_max))
        vector = encoder.encode_image(crop, config)
        embeddings.append(
            LanguageEmbedding(
                embedding_id=f"language-{region.region_id}",
                region_id=region.region_id,
                vector=vector,
                dimension=len(vector),
                model_id=config.backend,
                checkpoint=config.checkpoint,
                normalized=True,
            )
        )
    return tuple(embeddings)
