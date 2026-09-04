"""Merge determinístico de regions entre escalas (cross-scale).

Issue: #160.

Duas proposals só são tratadas como duplicatas da mesma region (e
mescladas) quando têm extensão quase idêntica: ou o IoU é alto, ou a
containment é alta em *ambos* os sentidos (mútua). Uma region pequena que
fica majoritariamente dentro de uma muito maior, sem o inverso ser
verdadeiro, é uma parte aninhada significativa, e é deliberadamente mantida
como sua própria region.
"""

from __future__ import annotations

import numpy as np

from visual_perception.config import RegionMergeConfig
from visual_perception.domain.geometry import Mask
from visual_perception.domain.identifiers import derive_region_id
from visual_perception.domain.regions import ObservedRegion, RegionProposal


# Ponto de entrada do merge: agrupa proposals duplicadas vindas de tiles/
# escalas diferentes (via union-find sobre IoU/containment) em regions
# canônicas estáveis. Existe para que o pipeline de tiling produza uma única
# region por objeto real, mesmo quando ele aparece em múltiplos tiles/escalas
# (issue #160); chamada depois da etapa de tiling no pipeline principal.
def merge_regions(
    observation_id: str,
    proposals: tuple[RegionProposal, ...],
    config: RegionMergeConfig,
) -> tuple[ObservedRegion, ...]:
    """Mescla proposals duplicadas entre tiles/escalas em regions canônicas estáveis."""
    ordered = sorted(proposals, key=lambda proposal: proposal.proposal_id)
    parent = {proposal.proposal_id: proposal.proposal_id for proposal in ordered}

    # Busca (find) da raiz do grupo union-find de uma proposal, com path
    # compression. Existe como parte interna do algoritmo union-find usado
    # para agrupar proposals duplicadas.
    def find(proposal_id: str) -> str:
        while parent[proposal_id] != proposal_id:
            parent[proposal_id] = parent[parent[proposal_id]]
            proposal_id = parent[proposal_id]
        return proposal_id

    # União (union) de dois grupos union-find, elegendo deterministicamente
    # a raiz de menor id. Existe como parte interna do algoritmo union-find;
    # chamada por merge_regions ao detectar que duas proposals são a mesma region.
    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    for i, left in enumerate(ordered):
        for right in ordered[i + 1 :]:
            iou = left.mask.iou(right.mask)
            mutual_containment = (
                left.mask.containment_ratio(right.mask) >= config.containment_merge_threshold
                and right.mask.containment_ratio(left.mask) >= config.containment_merge_threshold
            )
            if iou >= config.iou_merge_threshold or mutual_containment:
                union(left.proposal_id, right.proposal_id)

    groups: dict[str, list[RegionProposal]] = {}
    for proposal in ordered:
        groups.setdefault(find(proposal.proposal_id), []).append(proposal)

    regions = [_build_region(observation_id, group) for group in groups.values()]
    return tuple(sorted(regions, key=lambda region: region.region_id))


# Constrói a ObservedRegion final de um grupo de proposals já mescladas: une
# as masks (OR bit a bit), recalcula o box a partir da mask unida, e deriva
# um region_id estável a partir das proposals contribuintes. Chamada por
# merge_regions para cada grupo do union-find.
def _build_region(observation_id: str, group: list[RegionProposal]) -> ObservedRegion:
    group = sorted(group, key=lambda proposal: proposal.proposal_id)
    reference = group[0].mask
    merged_data = np.zeros_like(reference.data)
    for proposal in group:
        merged_data |= proposal.mask.data
    merged_mask = Mask(merged_data, reference.image_width, reference.image_height)
    contributing_ids = tuple(proposal.proposal_id for proposal in group)
    return ObservedRegion(
        region_id=derive_region_id(observation_id, contributing_ids),
        mask=merged_mask,
        box=merged_mask.bounding_box(),
        geometric_confidence=max(proposal.geometric_confidence for proposal in group),
        contributing_proposal_ids=contributing_ids,
    )
