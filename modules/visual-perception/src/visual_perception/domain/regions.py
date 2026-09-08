"""Contracts de proposta de região e de região observada canônica.

Issues: #158 (saída de fronteira de region discovery), #154/#160 (região
canônica), #193 (evidência multi-contexto por região).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from visual_perception.domain.geometry import BoundingBox, Mask
from visual_perception.domain.identifiers import validate_identifier
from visual_perception.domain.region_evidence import EvidenceSlot, RegionEvidenceSlot
from visual_perception.domain.semantics import ClaimKind, HypothesisRole, SemanticClaim


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


# Enumera por que uma proposta foi descartada antes de virar região. Existe
# como vocabulário fechado para que o motivo do descarte seja contável no
# diagnóstico, e não uma string livre por chamador: a #202 exige poder afirmar
# "nenhuma proposta sobrou fora da área válida", o que só é verificável se o
# motivo tiver identidade estável.
class ProposalRejectionReason(StrEnum):
    """Por que uma proposta foi descartada entre discovery e merge."""

    OUTSIDE_VALID_AREA = "outside_valid_area"
    EGO_VEHICLE_OVERLAP = "ego_vehicle_overlap"
    BELOW_MIN_RELATIVE_AREA = "below_min_relative_area"
    ABOVE_MAX_RELATIVE_AREA = "above_max_relative_area"


# Preserva a proveniência de uma proposta descartada. Existe porque descartar
# em silêncio destruiria a auditoria que este módulo promete: o artifact
# precisa poder responder *o que* foi removido, *por quê*, e *com que medida*.
@dataclass(frozen=True)
class RejectedProposal:
    """Uma proposta descartada, com o motivo e a medida que o justificou.

    Argumentos:
        proposal_id: identidade da proposta descartada.
        reason: motivo do descarte.
        value: a medida que disparou a regra (fração de sobreposição ou área
            relativa), preservada para que o limiar seja auditável.
        superseded_by: reservado para descartes que substituem uma proposta por
            outra; ``None`` para as regras de validade, que não têm substituta.
    """

    proposal_id: str
    reason: ProposalRejectionReason
    value: float
    superseded_by: str | None = None

    # Valida a identidade e a medida, para que um registro de descarte nunca
    # seja menos auditável que a proposta que ele substitui.
    def __post_init__(self) -> None:
        """Valida identidade, medida e a referência de redundância."""
        validate_identifier(self.proposal_id, field="proposal_id")
        if self.value < 0.0:
            raise ValueError(f"RejectedProposal.value must not be negative, got {self.value}.")
        if self.superseded_by is not None:
            validate_identifier(self.superseded_by, field="superseded_by")


# Representa uma região canônica e final dentro de uma VisualObservation, já
# depois de merge/refinamento. Existe como a unidade estável de região que o
# restante do sistema (sensor-association, semantic-fusion) consome.
@dataclass(frozen=True)
class ObservedRegion:
    """Uma região canônica e final dentro de uma :class:`VisualObservation`.

    ``geometric_confidence`` é distinta da confiança de qualquer claim
    semântico (ver #156): ela descreve só o quão confiável é a geometria da
    máscara/box.

    ``visual_embedding_ref``/``language_embedding_ref`` continuam sendo as
    referências canônicas de slot único. ``evidence`` (#193) acrescenta os
    slots multi-contexto complementares sem substituí-las: um consumidor que
    só entende o contract antigo continua funcionando.
    """

    region_id: str
    mask: Mask
    box: BoundingBox
    geometric_confidence: float
    contributing_proposal_ids: tuple[str, ...]
    claims: tuple[SemanticClaim, ...] = field(default_factory=tuple)
    visual_embedding_ref: str | None = None
    language_embedding_ref: str | None = None
    evidence: tuple[RegionEvidenceSlot, ...] = field(default_factory=tuple)

    # Valida o region_id, a confiança geométrica em [0, 1], que ao menos
    # uma proposta contribuinte foi preservada (para rastreabilidade até a
    # proveniência do merge), e que todo slot de evidência pertence a esta
    # região e não sobrescreve outro do mesmo slot/view.
    def __post_init__(self) -> None:
        """Valida identidade, confiança geométrica e coerência dos slots de evidência."""
        validate_identifier(self.region_id, field="region_id")
        if not 0.0 <= self.geometric_confidence <= 1.0:
            raise ValueError(
                f"geometric_confidence must be in [0, 1], got {self.geometric_confidence}."
            )
        if not self.contributing_proposal_ids:
            raise ValueError("ObservedRegion must preserve at least one contributing proposal id.")
        seen: set[tuple[EvidenceSlot, str | None]] = set()
        for slot in self.evidence:
            if slot.region_id != self.region_id:
                raise ValueError(
                    f"Evidence slot {slot.slot.value!r} references region {slot.region_id!r}, "
                    f"but belongs to {self.region_id!r}."
                )
            key = (slot.slot, slot.view_id)
            if key in seen:
                raise ValueError(
                    f"Region {self.region_id!r} has duplicate evidence for slot {slot.slot.value!r}"
                    f"{'' if slot.view_id is None else f' and view {slot.view_id!r}'}."
                )
            seen.add(key)

    # Busca um slot de evidência específico da região. Existe para que
    # consumidores (#194, #199) leiam foreground e contexto pelo nome do
    # slot, em vez de varrer a tupla em cada chamada.
    def evidence_for(
        self, slot: EvidenceSlot, *, view_id: str | None = None
    ) -> RegionEvidenceSlot | None:
        """Retorna o slot de evidência pedido, ou ``None`` se a região não o retém."""
        for item in self.evidence:
            if item.slot is slot and item.view_id == view_id:
                return item
        return None


# Retorna o claim de label primário de uma região: o primeiro claim de kind
# LABEL cujo papel é ``PRIMARY``. Existe para que overlay, diagnóstico e
# qualquer outro leitor contem exatamente o mesmo label — duas implementações
# da mesma política divergem em silêncio, e era exatamente o que acontecia
# quando esta função escolhia por posição e ``semantic_merge`` por score.
# Devolve o claim inteiro, e não só o texto, porque o consumidor quase sempre
# precisa também da confiança semântica, que é distinta da geométrica.
def primary_label_claim(region: ObservedRegion) -> SemanticClaim | None:
    """Retorna o claim de label primário da região, ou ``None`` se não houver.

    Argumentos:
        region: região observada cujos claims serão inspecionados.
    Retorna:
        o ``SemanticClaim`` de kind ``LABEL`` e papel ``PRIMARY``, ou ``None``
        quando a região não recebeu interpretação semântica.
    """
    for claim in region.claims:
        if claim.kind is ClaimKind.LABEL and claim.role is HypothesisRole.PRIMARY:
            return claim
    return None


# Lista as hipóteses de identidade concorrentes que o produtor registrou junto
# do primary. Existe para que o diagnóstico consiga medir hipóteses duplicadas
# e competição de identidade sem reimplementar o filtro por papel.
def alternative_label_claims(region: ObservedRegion) -> tuple[SemanticClaim, ...]:
    """Retorna os claims de label com papel ``ALTERNATIVE``, na ordem original.

    Argumentos:
        region: região observada cujos claims serão inspecionados.
    Retorna:
        as hipóteses concorrentes, ou uma tupla vazia quando não há nenhuma.
    """
    return tuple(
        claim
        for claim in region.claims
        if claim.kind is ClaimKind.LABEL and claim.role is HypothesisRole.ALTERNATIVE
    )
