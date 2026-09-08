"""Filtragem geométrica das proposals entre discovery e merge.

Issue: #202. Até esta etapa, ``run_canonical_pipeline`` promovia toda proposal
a região: os únicos filtros existentes — área mínima e score — viviam dentro do
adapter do SAM, e o estágio de aplicação não descartava nada. Medido em
``corridor-02-002``: 64 proposals viravam 60 regiões, das quais 7 caíam
majoritariamente fora do círculo útil da lente, 19 sobre o rig, e 48 pares
tinham containment acima de 0,8.

Este módulo decide **validade**: uma proposal entra em merge se descreve uma
parte utilizável do sensor, fora do rig, e com tamanho plausível para uma
região. Redundância geométrica continua sendo problema de ``region_merge.py``,
que une duplicatas preservando os dois ``contributing_proposal_ids`` — descartar
uma delas aqui perderia essa proveniência e deixaria o caminho de IoU do merge
inalcançável.

Não existe ``top-N`` aqui: um corte por número atingiria qualquer meta de
regiões descartando evidência útil junto com a inválida, e tornaria a métrica
autorrealizável.

Toda proposal descartada é preservada como :class:`RejectedProposal`, com o
motivo e a medida que o justificou, para que o artifact responda *o que* saiu e
*por quê*. E nada aqui toca nos pixels de origem: a exclusão opera sobre
máscaras, que é a regra estabelecida pela #212.
"""

from __future__ import annotations

from visual_perception.config import ProposalFilterConfig
from visual_perception.domain.geometry import Mask
from visual_perception.domain.image_area import ImageAreaMasks
from visual_perception.domain.regions import (
    ProposalRejectionReason,
    RegionProposal,
    RejectedProposal,
)

__all__ = ["filter_proposals"]


# Aplica todas as regras de exclusão a um conjunto de proposals globais.
# Chamada por run_canonical_pipeline logo depois do remapeamento de tile e
# antes do merge geométrico: em coordenadas globais, para que a comparação com
# as áreas do frame e entre proposals de tiles diferentes seja válida.
def filter_proposals(
    proposals: tuple[RegionProposal, ...],
    *,
    area_masks: ImageAreaMasks,
    config: ProposalFilterConfig,
) -> tuple[tuple[RegionProposal, ...], tuple[RejectedProposal, ...]]:
    """Separa as proposals que sobrevivem das que foram descartadas, com o motivo.

    Cada regra depende apenas da própria proposal, o que torna o resultado
    independente da ordem em que discovery as devolveu.

    Argumentos:
        proposals: as proposals já remapeadas para coordenadas globais.
        area_masks: as áreas declaradas do frame; ausentes desligam a exclusão
            por área, sem descartar nada em silêncio.
        config: os limiares de área e redundância.
    Retorna:
        as proposals mantidas, na ordem original, e os descartes registrados.
    """
    kept: list[RegionProposal] = []
    rejected: list[RejectedProposal] = []
    for proposal in proposals:
        rejection = _invalid_reason(proposal, area_masks=area_masks, config=config)
        if rejection is None:
            kept.append(proposal)
        else:
            rejected.append(rejection)
    return tuple(kept), tuple(rejected)


# Aplica as regras que dependem apenas da própria proposal. Isolada para que a
# ordem entre validade e redundância fique explícita em filter_proposals, e
# para que cada regra tenha um único ponto de decisão.
def _invalid_reason(
    proposal: RegionProposal, *, area_masks: ImageAreaMasks, config: ProposalFilterConfig
) -> RejectedProposal | None:
    """Retorna o descarte da proposal, ou ``None`` se ela é evidência válida."""
    area = proposal.mask.area()
    if area == 0:
        return None
    if area_masks.valid_area is not None:
        inside = _overlap_ratio(proposal.mask, area_masks.valid_area)
        if inside < config.min_valid_overlap:
            return RejectedProposal(
                proposal.proposal_id, ProposalRejectionReason.OUTSIDE_VALID_AREA, inside
            )
    if area_masks.ego_vehicle is not None:
        on_ego = _overlap_ratio(proposal.mask, area_masks.ego_vehicle)
        if on_ego > config.max_ego_overlap:
            return RejectedProposal(
                proposal.proposal_id, ProposalRejectionReason.EGO_VEHICLE_OVERLAP, on_ego
            )
    relative = area / (proposal.mask.image_width * proposal.mask.image_height)
    if relative < config.min_relative_area:
        return RejectedProposal(
            proposal.proposal_id, ProposalRejectionReason.BELOW_MIN_RELATIVE_AREA, relative
        )
    if relative > config.max_relative_area:
        return RejectedProposal(
            proposal.proposal_id, ProposalRejectionReason.ABOVE_MAX_RELATIVE_AREA, relative
        )
    return None


# Mede a fração de uma máscara que cai dentro de outra. Existe para que as
# regras de área e de containment usem exatamente a mesma razão, em vez de
# duas contas equivalentes que podem divergir em arredondamento.
def _overlap_ratio(mask: Mask, area: Mask) -> float:
    """Retorna a fração dos pixels de ``mask`` que caem dentro de ``area``."""
    total = mask.area()
    if total == 0:
        return 0.0
    return float((mask.data & area.data).sum()) / total
