"""Fronteira substituível para localização e segmentação de conceitos."""

from typing import Protocol

from visual_perception.config import SemanticGroundingConfig
from visual_perception.domain.grounding import GroundingPrediction, GroundingRequest
from visual_perception.domain.image_payload import ImagePayload


# Isola modelos caros do estágio determinístico de validação espacial e permite
# testes de integração com máscaras controladas, sem carregar GPU.
class SemanticGrounder(Protocol):
    """Produz uma predição por request no frame original, sem alterar discovery."""

    # Agrupa requests do mesmo frame para permitir compartilhar inferências e
    # carregar detector e segmentador sequencialmente em uma GPU pequena.
    def ground(
        self, image: ImagePayload, requests: tuple[GroundingRequest, ...], config: SemanticGroundingConfig
    ) -> tuple[GroundingPrediction, ...]:
        """Retorna máscaras condicionadas ao conceito ou falhas explícitas por região."""
        ...
