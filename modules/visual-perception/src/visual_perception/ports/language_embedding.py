"""Fronteira de embedding de região alinhado com linguagem.

Issue: #163.

Mantido separado de ``DenseFeatureExtractor`` (#161): um encoder alinhado com
linguagem produz uma segunda representação de região, independente,
alinhada com texto, que pode viver em um espaço diferente com um ciclo de
vida diferente.
"""

from __future__ import annotations

from typing import Protocol

from visual_perception.config import LanguageEmbeddingConfig
from visual_perception.domain.image_payload import ImagePayload


# Port que desacopla o pipeline do backend concreto de embedding
# visual-language (ex: CLIP, SigLIP). Existe porque a busca/consulta por
# vocabulário aberto depende de imagem e texto projetados no mesmo espaço,
# e esse espaço é específico do backend escolhido.
class LanguageAlignedEncoder(Protocol):
    """Codifica evidência de imagem de região e texto em um espaço compartilhado documentado."""

    # Projeta um crop de região no espaço de embedding compartilhado. Usado
    # pelo estágio de region semantics do pipeline para permitir busca por
    # texto sobre regiões descobertas.
    def encode_image(
        self, image: ImagePayload, config: LanguageEmbeddingConfig
    ) -> tuple[float, ...]:
        """Retorna um vetor de embedding finito para um crop de região."""
        ...

    # Projeta uma query de texto no mesmo espaço de embedding usado por
    # ``encode_image``, para permitir comparação/busca cross-modal.
    def encode_text(self, text: str, config: LanguageEmbeddingConfig) -> tuple[float, ...]:
        """Retorna um vetor de embedding finito para uma query de texto, quando suportado."""
        ...
