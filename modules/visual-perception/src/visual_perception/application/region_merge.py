"""Deterministic cross-scale region merge.

Issue: #160.

Two proposals are treated as duplicates of the same region (and merged) only
when they are near-identical in extent: either their IoU or their *mutual*
containment is high. A small region that sits mostly inside a much larger one
without the reverse being true is a meaningful nested part, and is
deliberately kept as its own region.
"""

from __future__ import annotations

import numpy as np

from visual_perception.config import RegionMergeConfig
from visual_perception.domain.geometry import Mask
from visual_perception.domain.identifiers import derive_region_id
from visual_perception.domain.regions import ObservedRegion, RegionProposal


def merge_regions(
    observation_id: str,
    proposals: tuple[RegionProposal, ...],
    config: RegionMergeConfig,
) -> tuple[ObservedRegion, ...]:
    """Merge duplicate proposals across tiles/scales into stable canonical regions."""
    ordered = sorted(proposals, key=lambda proposal: proposal.proposal_id)
    parent = {proposal.proposal_id: proposal.proposal_id for proposal in ordered}

    def find(proposal_id: str) -> str:
        while parent[proposal_id] != proposal_id:
            parent[proposal_id] = parent[parent[proposal_id]]
            proposal_id = parent[proposal_id]
        return proposal_id

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
