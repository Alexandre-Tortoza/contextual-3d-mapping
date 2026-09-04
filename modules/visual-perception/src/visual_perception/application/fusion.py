"""Fusão de evidência de percepção multi-fonte.

Issue: #182.

A geometria de região de múltiplas fontes é fundida reaproveitando a
mesma lógica de overlap/containment do merge single-source (#160): uma
fonte é apenas mais um contribuidor para
:func:`~visual_perception.application.region_merge.merge_regions`, e cada
proposal contribuinte mantém seu próprio label ``source``, então a
proveniência continua recuperável depois da fusão.

Claims semânticos são fundidos apenas quando as fontes *concordam* (mesmo
kind e value): claims concordantes colapsam em um único claim com
confiança calibrada. Claims discordantes são deliberadamente deixados
como claims separados e coexistentes — a detecção de contradição já
existente no quality auditor (#168) é o que evidencia essa discordância,
em vez de um segundo mecanismo duplicado aqui.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict

from visual_perception.application.region_merge import merge_regions
from visual_perception.config import RegionMergeConfig
from visual_perception.domain.regions import ObservedRegion, RegionProposal
from visual_perception.domain.semantics import ConfidenceScore, SemanticClaim


# Funde region proposals contribuídos por duas ou mais fontes em regiões
# consolidadas, recalculando a confiança geométrica por peso de fonte
# quando mais de uma fonte contribui para a mesma região final.
def fuse_proposals(
    proposals_by_source: dict[str, tuple[RegionProposal, ...]],
    observation_id: str,
    merge_config: RegionMergeConfig,
    source_calibration: dict[str, float] | None = None,
) -> tuple[ObservedRegion, ...]:
    """Funde region proposals contribuídos por duas ou mais fontes.

    ``source_calibration`` mapeia um nome de fonte para um peso de
    confiança usado para combinar a confiança geométrica quando mais de
    uma fonte contribui para a mesma região final; fontes ausentes do
    mapeamento assumem 1.0 por padrão.
    """
    all_proposals = [proposal for proposals in proposals_by_source.values() for proposal in proposals]
    proposals_by_id = {proposal.proposal_id: proposal for proposal in all_proposals}
    regions = merge_regions(observation_id, tuple(all_proposals), merge_config)

    calibration = source_calibration or {}
    fused_regions = []
    for region in regions:
        contributing = [proposals_by_id[proposal_id] for proposal_id in region.contributing_proposal_ids]
        sources = {proposal.source for proposal in contributing}
        if len(sources) > 1:
            weights = [calibration.get(proposal.source, 1.0) for proposal in contributing]
            total_weight = sum(weights)
            weighted = zip(weights, contributing, strict=True)
            confidence = (
                sum(weight * proposal.geometric_confidence for weight, proposal in weighted) / total_weight
                if total_weight > 0
                else region.geometric_confidence
            )
            region = dataclasses.replace(region, geometric_confidence=confidence)
        fused_regions.append(region)
    return tuple(fused_regions)


# Consolida claims semânticos concordantes entre fontes por região; chamada
# depois de fuse_proposals para produzir o conjunto final de claims por
# região usado pelo pipeline canônico.
def fuse_claims(regions: tuple[ObservedRegion, ...]) -> tuple[ObservedRegion, ...]:
    """Colapsa em um só claims de mesmo kind e mesmo value vindos de fontes diferentes.

    Claims que discordam em value são deixados intocados e coexistem, para
    que a detecção de contradição do quality auditor continue se aplicando
    a eles.
    """
    return tuple(dataclasses.replace(region, claims=_fuse_region_claims(region.claims)) for region in regions)


# Agrupa os claims de uma região por (kind, value) e colapsa cada grupo
# concordante em um único claim com confiança média e evidência combinada;
# helper interno de fuse_claims.
def _fuse_region_claims(claims: tuple[SemanticClaim, ...]) -> tuple[SemanticClaim, ...]:
    groups: dict[tuple[str, str], list[SemanticClaim]] = defaultdict(list)
    for claim in claims:
        groups[(claim.kind.value, claim.value)].append(claim)

    fused: list[SemanticClaim] = []
    for group in groups.values():
        if len(group) == 1:
            fused.append(group[0])
            continue
        sources = sorted({claim.provenance.producer for claim in group})
        average_confidence = sum(claim.confidence.value for claim in group) / len(group)
        combined_evidence = tuple(evidence for claim in group for evidence in claim.evidence)
        fused.append(
            dataclasses.replace(
                group[0],
                confidence=ConfidenceScore(average_confidence, source=f"fused({','.join(sources)})"),
                evidence=combined_evidence,
            )
        )
    return tuple(fused)
