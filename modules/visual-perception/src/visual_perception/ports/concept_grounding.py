"""Fronteira de grounding de conceitos textuais em máscaras (#277)."""

from __future__ import annotations

from typing import Protocol

from visual_perception.config import ConceptGroundingConfig
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.regions import LocalRegionProposal


# Port separado do RegionDiscoverer porque recebe conceitos e produz propostas
# rastreáveis ao conceito; o discovery genérico continua sem entrada textual.
class ConceptRegionDiscoverer(Protocol):
    """Localiza e segmenta cada conceito pedido na imagem recebida."""

    # Devolve propostas locais com ``concept`` preenchido. Uma implementação deve
    # reaproveitar as features da imagem entre conceitos quando o backend permite.
    def discover_concept_regions(
        self, image: ImagePayload, concepts: tuple[str, ...], config: ConceptGroundingConfig
    ) -> tuple[LocalRegionProposal, ...]:
        """Retorna as propostas de todos os conceitos, ordenadas por conceito e score."""
        ...
