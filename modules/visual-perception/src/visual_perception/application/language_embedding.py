"""Stage de embedding de região alinhado a linguagem.

Issue: #163.

NOTE: o pipeline canônico passou a obter esse embedding através de
``application/multi_context.py`` (#194), que produz o mesmo crop justo como
um slot de evidência ao lado dos demais. Esta função continua pública como o
caminho de slot único, para consumidores que só precisam do embedding de
região e não da estrutura multi-contexto.
"""

from __future__ import annotations

from visual_perception.config import LanguageEmbeddingConfig
from visual_perception.domain.embeddings import LanguageEmbedding
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.regions import ObservedRegion
from visual_perception.ports.language_embedding import LanguageAlignedEncoder


# Codifica o crop de cada região no espaço de embedding alinhado a
# linguagem, usada pelo pipeline canônico logo após o pooling de features
# visuais para permitir consultas semânticas por texto sobre as regiões.
def encode_regions(
    regions: tuple[ObservedRegion, ...],
    image: ImagePayload,
    encoder: LanguageAlignedEncoder,
    config: LanguageEmbeddingConfig,
) -> tuple[LanguageEmbedding, ...]:
    """Codifica o crop de cada região no espaço de embedding alinhado a linguagem."""
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
