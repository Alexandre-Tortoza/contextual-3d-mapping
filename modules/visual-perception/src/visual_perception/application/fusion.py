"""Multi-source perception evidence fusion.

Issue: #182.

Region geometry from multiple sources is fused by reusing the same
overlap/containment logic as single-source merge (#160): a source is just
another contributor to :func:`~visual_perception.application.region_merge.merge_regions`,
and each contributing proposal keeps its own ``source`` label, so provenance
stays recoverable after fusion.

Semantic claims are fused only when sources *agree* (same kind and value):
agreeing claims collapse into one claim with a calibrated confidence.
Disagreeing claims are deliberately left as separate, coexisting claims —
the existing contradiction detection in the quality auditor (#168) is what
surfaces that disagreement, rather than a second, duplicate mechanism here.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict

from visual_perception.application.region_merge import merge_regions
from visual_perception.config import RegionMergeConfig
from visual_perception.domain.regions import ObservedRegion, RegionProposal
from visual_perception.domain.semantics import ConfidenceScore, SemanticClaim


def fuse_proposals(
    proposals_by_source: dict[str, tuple[RegionProposal, ...]],
    observation_id: str,
    merge_config: RegionMergeConfig,
    source_calibration: dict[str, float] | None = None,
) -> tuple[ObservedRegion, ...]:
    """Merge region proposals contributed by two or more sources.

    ``source_calibration`` maps a source name to a trust weight used to
    combine geometric confidence when more than one source contributes to
    the same final region; sources missing from the mapping default to 1.0.
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


def fuse_claims(regions: tuple[ObservedRegion, ...]) -> tuple[ObservedRegion, ...]:
    """Collapse same-kind, same-value claims from different sources into one.

    Claims that disagree in value are left untouched and coexist, so the
    quality auditor's contradiction detection still applies to them.
    """
    return tuple(dataclasses.replace(region, claims=_fuse_region_claims(region.claims)) for region in regions)


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
