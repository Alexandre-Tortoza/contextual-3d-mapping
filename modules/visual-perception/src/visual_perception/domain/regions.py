"""Contracts de proposta de região e de região observada canônica.

Issues: #158 (saída de fronteira de region discovery), #154/#160 (região canônica).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from visual_perception.domain.geometry import BoundingBox, Mask
from visual_perception.domain.identifiers import validate_identifier
from visual_perception.domain.semantics import SemanticClaim


# Registra qual escala/tile de uma passada (possivelmente tiled,
# possivelmente multi-escala) produziu uma proposta. Existe para que uma
# proposta local possa ser remapeada de volta ao seu tile/escala de origem
# (#159).
@dataclass(frozen=True)
class TileProvenance:
    """Qual escala/tile de uma passada (possivelmente tiled, possivelmente
    multi-escala) produziu uma proposta.

    Issue: #159.
    """

    scale_id: str
    tile_id: str

    # Valida que scale_id e tile_id são identificadores bem formados.
    def __post_init__(self) -> None:
        validate_identifier(self.scale_id, field="scale_id")
        validate_identifier(self.tile_id, field="tile_id")


# Representa uma região candidata em coordenadas de pixel locais a um tile,
# como produzida diretamente por um RegionDiscoverer antes do remapeamento
# para a imagem original. Existe como o formato intermediário, pré-remapeamento,
# entre discovery e o restante do pipeline (#159).
@dataclass(frozen=True)
class LocalRegionProposal:
    """Uma região candidata em coordenadas de pixel locais a um tile, como
    produzida diretamente por um :class:`~visual_perception.ports.region_discovery.RegionDiscoverer`
    antes do remapeamento para a imagem original (ver #159).
    """

    local_id: str
    mask: Mask
    box: BoundingBox
    geometric_confidence: float
    source: str

    # Valida o id local, a confiança geométrica em [0, 1], a presença de
    # source, e que a máscara não está vazia.
    def __post_init__(self) -> None:
        validate_identifier(self.local_id, field="local_id")
        if not 0.0 <= self.geometric_confidence <= 1.0:
            raise ValueError(
                f"geometric_confidence must be in [0, 1], got {self.geometric_confidence}."
            )
        if not self.source:
            raise ValueError("source must not be empty.")
        if self.mask.is_empty:
            raise ValueError(f"LocalRegionProposal({self.local_id!r}) has an empty mask.")


# Representa uma região candidata como produzida por um backend de region
# discovery, já remapeada para coordenadas da imagem original. Existe como a
# proposta pronta para merge/refinamento, depois que a fronteira de tiling
# (#159) já remapeou as coordenadas locais.
@dataclass(frozen=True)
class RegionProposal:
    """Uma região candidata como produzida por um backend de region
    discovery, já remapeada para coordenadas da imagem original (ver a
    fronteira de tiling da #159).
    """

    proposal_id: str
    mask: Mask
    box: BoundingBox
    geometric_confidence: float
    source: str
    tile: TileProvenance

    # Valida o id da proposta, a confiança geométrica em [0, 1], a presença
    # de source, e que a máscara não está vazia.
    def __post_init__(self) -> None:
        validate_identifier(self.proposal_id, field="proposal_id")
        if not 0.0 <= self.geometric_confidence <= 1.0:
            raise ValueError(
                f"geometric_confidence must be in [0, 1], got {self.geometric_confidence}."
            )
        if not self.source:
            raise ValueError("source must not be empty.")
        if self.mask.is_empty:
            raise ValueError(f"RegionProposal({self.proposal_id!r}) has an empty mask.")


# Representa uma região canônica e final dentro de uma VisualObservation, já
# depois de merge/refinamento. Existe como a unidade estável de região que o
# restante do sistema (sensor-association, semantic-fusion) consome.
@dataclass(frozen=True)
class ObservedRegion:
    """Uma região canônica e final dentro de uma :class:`VisualObservation`.

    ``geometric_confidence`` é distinta da confiança de qualquer claim
    semântico (ver #156): ela descreve só o quão confiável é a geometria da
    máscara/box.
    """

    region_id: str
    mask: Mask
    box: BoundingBox
    geometric_confidence: float
    contributing_proposal_ids: tuple[str, ...]
    claims: tuple[SemanticClaim, ...] = field(default_factory=tuple)
    visual_embedding_ref: str | None = None
    language_embedding_ref: str | None = None

    # Valida o region_id, a confiança geométrica em [0, 1], e que ao menos
    # uma proposta contribuinte foi preservada (para rastreabilidade até a
    # proveniência do merge).
    def __post_init__(self) -> None:
        validate_identifier(self.region_id, field="region_id")
        if not 0.0 <= self.geometric_confidence <= 1.0:
            raise ValueError(
                f"geometric_confidence must be in [0, 1], got {self.geometric_confidence}."
            )
        if not self.contributing_proposal_ids:
            raise ValueError("ObservedRegion must preserve at least one contributing proposal id.")
