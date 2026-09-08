"""Testes de contract do modelo de suporte semântico calibrado (#195)."""

from __future__ import annotations

import json

import pytest

from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.semantic_support import (
    SemanticSupport,
    SupportEvidence,
    SupportInputs,
    SupportPolarity,
    SupportState,
    semantic_support_from_dict,
    semantic_support_to_dict,
)
from visual_perception.domain.semantics import (
    ClaimKind,
    ConfidenceScore,
    Evidence,
    HypothesisRole,
    SemanticClaim,
    calibrated_confidence_of,
    most_confident_claim,
    most_supported_claim,
)

_PROVENANCE = ModelProvenance(stage="region_semantics", producer="vlm", config_fingerprint="abc123")
_EVIDENCE = (Evidence(description="raw region response"),)


# Constrói uma claim de label com o suporte pedido, para que os testes
# variem apenas o que estão medindo.
def _claim(value: str, raw: float | None = None, support: SemanticSupport | None = None) -> SemanticClaim:
    confidence = None if raw is None else ConfidenceScore(raw, source="vlm")
    return SemanticClaim(
        ClaimKind.LABEL,
        value,
        confidence,
        _EVIDENCE,
        _PROVENANCE,
        support=support,
        role=HypothesisRole.PRIMARY,
    )


# Constrói um suporte calibrado válido com o valor pedido.
def _calibrated(value: float) -> SemanticSupport:
    return SemanticSupport(
        state=SupportState.CALIBRATED,
        calibrated_confidence=ConfidenceScore(value, source="calibrated:vlm"),
        calibration_version="calibration/1",
        visual_support=0.9,
    )


def test_an_unscored_source_is_represented_without_inventing_a_number() -> None:
    claim = _claim("door", raw=None, support=SemanticSupport(state=SupportState.UNKNOWN))

    assert claim.confidence is None
    assert claim.support.state is SupportState.UNKNOWN
    assert claim.support.calibrated_confidence is None
    assert calibrated_confidence_of(claim) is None


def test_raw_and_calibrated_scores_cannot_be_confused_in_public_output() -> None:
    raw_only = _claim("door", raw=0.9)
    calibrated = _claim("window", raw=0.4, support=_calibrated(0.85))

    claims = (raw_only, calibrated)

    assert most_confident_claim(claims) is raw_only
    assert most_supported_claim(claims) is calibrated
    assert calibrated.confidence.value == 0.4
    assert calibrated_confidence_of(calibrated).value == 0.85


def test_only_the_calibrated_state_may_carry_a_calibrated_score() -> None:
    for state in (SupportState.UNKNOWN, SupportState.RAW):
        with pytest.raises(ValueError, match="must not carry a calibrated_confidence"):
            SemanticSupport(
                state=state, calibrated_confidence=ConfidenceScore(0.5, source="calibrated:vlm")
            )
    with pytest.raises(ValueError, match="must carry a calibrated_confidence"):
        SemanticSupport(state=SupportState.CALIBRATED, calibration_version="calibration/1")


def test_abstention_and_failure_must_explain_themselves() -> None:
    for state in (SupportState.ABSTAINED, SupportState.FAILED):
        with pytest.raises(ValueError, match="must explain itself"):
            SemanticSupport(state=state)
    assert SemanticSupport(state=SupportState.ABSTAINED, reason="insufficient support").reason


def test_invalid_confidence_ranges_and_signals_fail_validation() -> None:
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        ConfidenceScore(1.5, source="vlm")
    with pytest.raises(ValueError, match="must be a real number"):
        ConfidenceScore(True, source="vlm")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"visual_support must be in \[0, 1\]"):
        SemanticSupport(state=SupportState.RAW, visual_support=1.2)
    with pytest.raises(ValueError, match="region_quality must be a real number"):
        SemanticSupport(state=SupportState.RAW, region_quality=True)  # type: ignore[arg-type]


def test_an_unsupported_calibration_version_fails_validation() -> None:
    with pytest.raises(ValueError, match="Unsupported calibration_version"):
        SemanticSupport(
            state=SupportState.CALIBRATED,
            calibrated_confidence=ConfidenceScore(0.5, source="calibrated:vlm"),
            calibration_version="calibration/99",
        )


def test_missing_required_provenance_fails_validation() -> None:
    with pytest.raises(ValueError, match="must record its calibration_version"):
        SemanticSupport(
            state=SupportState.CALIBRATED,
            calibrated_confidence=ConfidenceScore(0.5, source="calibrated:vlm"),
        )
    with pytest.raises(ValueError, match="artifact_uri must not be empty"):
        SupportEvidence(artifact_uri="", source="vlm", polarity=SupportPolarity.SUPPORTIVE)


def test_a_claim_preserves_supportive_and_contradictory_evidence_side_by_side() -> None:
    support = SemanticSupport(
        state=SupportState.RAW,
        evidence=(
            SupportEvidence(
                "visual-region-a", "dinov2", SupportPolarity.SUPPORTIVE, "region-a", "tight_crop"
            ),
            SupportEvidence("claim:label:window", "vlm", SupportPolarity.CONTRADICTORY, "region-a"),
        ),
    )

    assert len(support.evidence_with(SupportPolarity.SUPPORTIVE)) == 1
    assert len(support.evidence_with(SupportPolarity.CONTRADICTORY)) == 1


def test_freely_generated_claims_may_never_carry_a_raw_confidence() -> None:
    for kind in (ClaimKind.ATTRIBUTE, ClaimKind.CONDITION, ClaimKind.MATERIAL, ClaimKind.HAZARD):
        with pytest.raises(ValueError, match="must not carry a raw confidence"):
            SemanticClaim(kind, "wet", ConfidenceScore(1.0, source="vlm"), _EVIDENCE, _PROVENANCE)
    # Elas continuam pontuáveis pelo caminho calibrado, que é baseado em evidência.
    scored = SemanticClaim(ClaimKind.HAZARD, "wet floor", None, _EVIDENCE, _PROVENANCE, _calibrated(0.7))
    assert calibrated_confidence_of(scored).value == 0.7


def test_the_raw_model_response_stays_auditable_next_to_the_claim() -> None:
    evidence = Evidence(description="raw", raw_response_json=json.dumps({"label": "door"}))

    assert json.loads(evidence.raw_response_json)["label"] == "door"
    with pytest.raises(ValueError, match="must be valid JSON"):
        Evidence(description="raw", raw_response_json="{not json")
    with pytest.raises(ValueError, match="must decode to a JSON object"):
        Evidence(description="raw", raw_response_json="[1, 2]")


def test_support_round_trips_through_serialization() -> None:
    support = SemanticSupport(
        state=SupportState.ABSTAINED,
        reason="insufficient support: visual support not measured",
        source_reliability=0.4,
        visual_support=None,
        region_quality=0.8,
        contradiction_support=0.5,
        evidence=(SupportEvidence("visual-region-a", "dinov2", SupportPolarity.SUPPORTIVE),),
        domain="indoor_corridor",
        calibration_version="calibration/1",
        calibration_artifact="deadbeef",
    )

    assert semantic_support_from_dict(semantic_support_to_dict(support)) == support
    assert semantic_support_to_dict(None) is None
    assert semantic_support_from_dict(None) is None


def test_support_inputs_reject_out_of_range_signals_at_the_calibration_boundary() -> None:
    with pytest.raises(ValueError, match="claim_kind must not be empty"):
        SupportInputs(claim_kind="", source="vlm")
    with pytest.raises(ValueError, match=r"contradiction_support must be in \[0, 1\]"):
        SupportInputs(claim_kind="label", source="vlm", contradiction_support=-0.1)
