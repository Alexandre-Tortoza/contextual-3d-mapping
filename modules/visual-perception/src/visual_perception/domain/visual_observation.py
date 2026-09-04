"""Contract de saída canônico de observação visual.

Issue: #154.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from visual_perception.domain.references import ObservationReference
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.relations import CandidateRelation, validate_relation_references
from visual_perception.domain.semantics import SemanticClaim

#: Convenção de coordenadas de imagem vinculante para toda VisualObservation.
#: Ver ``domain/geometry.py`` para a definição completa.
COORDINATE_CONVENTION = "top-left-origin,half-open-xyxy"


# Agrupa claims semânticos de nível de cena que não têm origem na geometria
# de nenhuma região específica. Existe separado de ObservedRegion.claims
# porque descreve a cena como um todo (#164), não uma região individual.
@dataclass(frozen=True)
class SceneContext:
    """Claims semânticos de nível de cena que não são a origem da geometria de nenhuma região.

    Issue: #164.
    """

    claims: tuple[SemanticClaim, ...] = field(default_factory=tuple)


# Representa a saída completa e canônica da percepção visual para uma
# imagem. Existe como o contract final que sensor-association e os módulos
# downstream consomem, agregando cena, regiões e relações em um único
# objeto auditável.
@dataclass(frozen=True)
class VisualObservation:
    """A saída completa e canônica da percepção visual para uma imagem.

    Identidade e timing vivem em ``source`` (a ``ObservationReference``
    compartilhada, #100), sem duplicação como campos separados aqui.
    """

    source: ObservationReference
    image_width: int
    image_height: int
    scene_context: SceneContext
    regions: tuple[ObservedRegion, ...]
    relations: tuple[CandidateRelation, ...]
    schema_version: int = 1
    coordinate_convention: str = COORDINATE_CONVENTION

    # Valida a resolução da imagem, a unicidade de region_id entre as
    # regiões, que a resolução de cada máscara de região bate com a da
    # observação, e que toda relação referencia apenas regiões conhecidas.
    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image_width and image_height must be positive.")
        region_ids = [region.region_id for region in self.regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("VisualObservation.regions must have unique region_id values.")
        for region in self.regions:
            if (region.mask.image_width, region.mask.image_height) != (
                self.image_width,
                self.image_height,
            ):
                raise ValueError(
                    f"ObservedRegion({region.region_id!r}) mask resolution does not match "
                    "the observation's image resolution."
                )
        validate_relation_references(self.relations, frozenset(region_ids))

    # Expõe o observation_id a partir de source, sem duplicar o campo no
    # próprio VisualObservation.
    @property
    def observation_id(self) -> str:
        return str(self.source.observation_id)

    # Busca uma região pelo seu region_id. Usada por consumidores que
    # recebem um region_id (ex: de uma relação) e precisam do objeto
    # ObservedRegion completo.
    def region_by_id(self, region_id: str) -> ObservedRegion:
        for region in self.regions:
            if region.region_id == region_id:
                return region
        raise KeyError(f"No region with region_id={region_id!r}.")
