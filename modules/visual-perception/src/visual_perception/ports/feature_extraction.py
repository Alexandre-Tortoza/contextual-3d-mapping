"""Fronteira de extração de features visuais densas.

Issue: #161.
"""

from __future__ import annotations

from typing import Protocol

from visual_perception.config import FeatureExtractionConfig
from visual_perception.domain.feature_map import FeatureMap
from visual_perception.domain.image_payload import ImagePayload


# Port (Protocol) que desacopla o pipeline de percepção visual do backend
# concreto de extração de features (ex: DINOv2, CLIP visual encoder). Existe
# para que o backend real de GPU e o fake usado em testes satisfaçam a mesma
# fronteira, sem vazar tipos de tensor/modelo para as camadas de domain e
# application.
class DenseFeatureExtractor(Protocol):
    """Extrai um grid espacial de features de uma imagem ou crop."""

    # Ponto de entrada único do port: recebe uma imagem/crop e devolve um
    # feature map espacial. Chamado pelo estágio de feature extraction do
    # pipeline (application/pipeline.py) a cada região/tile.
    def extract(self, image: ImagePayload, config: FeatureExtractionConfig) -> FeatureMap:
        """Retorna um feature map denso e finito, alinhado a ``image``."""
        ...
