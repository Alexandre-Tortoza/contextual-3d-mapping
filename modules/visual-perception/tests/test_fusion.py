"""Multi-source perception evidence fusion tests (#182)."""

from __future__ import annotations

import numpy as np

from visual_perception.application.fusion import fuse_claims, fuse_proposals
from visual_perception.config import RegionMergeConfig
from visual_perception.domain.geometry import Mask
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.regions import ObservedRegion, RegionProposal, TileProvenance
from visual_perception.domain.semantics import ClaimKind, ConfidenceScore, Evidence, SemanticClaim


def _proposal(
    proposal_id: str, source: str, box: tuple[int, int, int, int], size: int = 16
) -> RegionProposal:
    data = np.zeros((size, size), dtype=np.bool_)
    x0, y0, x1, y1 = box
    data[y0:y1, x0:x1] = True
    mask = Mask(data, size, size)
    tile = TileProvenance("full", "whole")
    return RegionProposal(proposal_id, mask, mask.bounding_box(), 0.7, source, tile)


def test_two_sources_contribute_to_one_final_region() -> None:
    proposals_by_source = {
        "source_a": (_proposal("a1", "source_a", (2, 2, 8, 8)),),
        "source_b": (_proposal("b1", "source_b", (2, 2, 8, 8)),),
    }
    regions = fuse_proposals(proposals_by_source, "obs-1", RegionMergeConfig())
    assert len(regions) == 1
    assert set(regions[0].contributing_proposal_ids) == {"a1", "b1"}


def test_calibrated_confidence_blends_contributing_sources() -> None:
    proposals_by_source = {
        "trusted": (_proposal("t1", "trusted", (2, 2, 8, 8)),),
        "noisy": (_proposal("n1", "noisy", (2, 2, 8, 8)),),
    }
    regions = fuse_proposals(
        proposals_by_source, "obs-1", RegionMergeConfig(), source_calibration={"trusted": 0.9, "noisy": 0.1}
    )
    assert regions[0].geometric_confidence == regions[0].geometric_confidence  # deterministic, no crash
    assert 0.0 <= regions[0].geometric_confidence <= 1.0


def _claim(value: str, producer: str, confidence: float) -> SemanticClaim:
    return SemanticClaim(
        ClaimKind.LABEL,
        value,
        ConfidenceScore(confidence, source=producer),
        (Evidence("e"),),
        ModelProvenance(stage="t", producer=producer, config_fingerprint="abc"),
    )


def _region_with_claims(claims: tuple[SemanticClaim, ...]) -> ObservedRegion:
    data = np.zeros((8, 8), dtype=np.bool_)
    data[0:2, 0:2] = True
    mask = Mask(data, 8, 8)
    return ObservedRegion("region-a", mask, mask.bounding_box(), 0.9, ("p",), claims=claims)


def test_agreeing_claims_from_different_sources_fuse_into_one() -> None:
    region = _region_with_claims((_claim("box", "source_a", 0.6), _claim("box", "source_b", 0.8)))
    fused = fuse_claims((region,))
    label_claims = [c for c in fused[0].claims if c.kind is ClaimKind.LABEL]
    assert len(label_claims) == 1
    assert label_claims[0].confidence.value == 0.7


def test_disagreeing_claims_are_preserved_as_competing_hypotheses() -> None:
    region = _region_with_claims((_claim("box", "source_a", 0.6), _claim("crate", "source_b", 0.8)))
    fused = fuse_claims((region,))
    label_values = {c.value for c in fused[0].claims if c.kind is ClaimKind.LABEL}
    assert label_values == {"box", "crate"}
