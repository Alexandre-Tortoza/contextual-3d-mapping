"""Verifica compatibilidade de contract de saída com sensor-association.

Issue: #178.

`sensor-association` (#137) precisa, para cada região: uma identidade
estável, geometria em espaço de imagem (mask/box) sob a convenção de
coordenadas documentada, o timing e a identidade de frame da observação, e
evidência/proveniência. Este módulo extrai exatamente essa fatia usando
apenas contracts públicos, para que um campo ausente falhe nos testes de
integração em vez de causar uma reinterpretação silenciosa.
"""

from __future__ import annotations

from dataclasses import dataclass

from contextual_mapping_contracts import FrameId, Timestamp

from visual_perception.domain.geometry import BoundingBox, Mask
from visual_perception.domain.visual_observation import VisualObservation


# Representa a fatia mínima de uma região que sensor-association consegue
# consumir — só os campos que esse módulo consumidor realmente precisa,
# extraídos via contracts públicos, nunca via acesso interno a VisualObservation.
@dataclass(frozen=True)
class RegionAssociationFixture:
    """A fatia mínima de uma região que sensor-association consegue consumir."""

    observation_id: str
    region_id: str
    mask: Mask
    box: BoundingBox
    timestamp: Timestamp
    frame_id: FrameId
    coordinate_convention: str


# Constrói um RegionAssociationFixture por região da observação, usando
# somente os campos públicos de VisualObservation — serve de fixture de teste
# de integração para verificar que sensor-association consegue consumir a
# saída deste módulo sem depender de detalhes internos.
def build_association_fixtures(observation: VisualObservation) -> tuple[RegionAssociationFixture, ...]:
    """Constrói um fixture por região, usando apenas os campos públicos de ``VisualObservation``."""
    return tuple(
        RegionAssociationFixture(
            observation_id=observation.observation_id,
            region_id=region.region_id,
            mask=region.mask,
            box=region.box,
            timestamp=observation.source.timestamp,
            frame_id=observation.source.frame_id,
            coordinate_convention=observation.coordinate_convention,
        )
        for region in observation.regions
    )
