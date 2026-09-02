"""Semantic claims, confidence, evidence, and provenance contract tests (#156)."""

from __future__ import annotations

import pytest

from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.semantics import (
    ClaimKind,
    ConfidenceScore,
    Evidence,
    SemanticClaim,
    contradicting_claims,
)


def _provenance() -> ModelProvenance:
    return ModelProvenance(stage="test", producer="fake", config_fingerprint="abc123")


def _claim(kind: ClaimKind, value: str, confidence: float = 0.9) -> SemanticClaim:
    return SemanticClaim(
        kind, value, ConfidenceScore(confidence, source="fake"), (Evidence("evidence"),), _provenance()
    )


def test_confidence_score_rejects_out_of_range_value() -> None:
    with pytest.raises(ValueError):
        ConfidenceScore(1.5, source="fake")


def test_claim_requires_at_least_one_evidence() -> None:
    with pytest.raises(ValueError):
        SemanticClaim(ClaimKind.LABEL, "box", ConfidenceScore(0.9, source="fake"), (), _provenance())


def test_single_claim_has_no_contradiction() -> None:
    claims = (_claim(ClaimKind.LABEL, "box"),)
    assert contradicting_claims(claims, ClaimKind.LABEL) == ()


def test_multiple_agreeing_claims_have_no_contradiction() -> None:
    claims = (_claim(ClaimKind.LABEL, "box"), _claim(ClaimKind.LABEL, "box", 0.5))
    assert contradicting_claims(claims, ClaimKind.LABEL) == ()


def test_contradictory_hypotheses_coexist_and_are_detected() -> None:
    claims = (_claim(ClaimKind.LABEL, "box"), _claim(ClaimKind.LABEL, "crate"))
    contradictions = contradicting_claims(claims, ClaimKind.LABEL)
    assert len(contradictions) == 2
    assert {claim.value for claim in contradictions} == {"box", "crate"}
