"""Fronteira de descoberta de conceitos visuais concretos em nível de cena (#277)."""

from __future__ import annotations

from typing import Any, Protocol

from visual_perception.config import MultimodalReasoningConfig
from visual_perception.domain.image_payload import ImagePayload


# Port separado do MultimodalReasoner porque o consumidor é outro (o grounding por
# conceito) e a capacidade pode ser desligada sem afetar cena e região. Os
# adapters de VLM satisfazem os dois ports com o mesmo modelo residente.
class SceneConceptDiscoverer(Protocol):
    """Propõe conceitos concretos a procurar no frame."""

    # Devolve a resposta estruturada bruta; normalização, exclusões e teto são
    # decisões de application (scene_concept_discovery.py).
    def discover_concepts(
        self, image: ImagePayload, config: MultimodalReasoningConfig, *, max_concepts: int
    ) -> dict[str, Any]:
        """Retorna ``{"entities": [...], "contextual_features": [...]}`` bruto do backend."""
        ...
