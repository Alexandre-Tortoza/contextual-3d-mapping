"""Testes de contract de claims semânticos, confidence, evidence e provenance (#156)."""

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


# Provenance mínima e reutilizável para os SemanticClaims construídos nos testes
# abaixo — isola o teste de detalhes de proveniência que não são o foco do caso.
def _provenance() -> ModelProvenance:
    return ModelProvenance(stage="test", producer="fake", config_fingerprint="abc123")


# Constrói um SemanticClaim mínimo válido (kind/value/confidence/evidence/provenance),
# usado como bloco básico pelos testes de contradicting_claims.
def _claim(kind: ClaimKind, value: str, confidence: float = 0.9) -> SemanticClaim:
    return SemanticClaim(
        kind, value, ConfidenceScore(confidence, source="fake"), (Evidence("evidence"),), _provenance()
    )


# ConfidenceScore deve rejeitar valores fora do intervalo [0, 1] válido para uma
# confiança.
def test_confidence_score_rejects_out_of_range_value() -> None:
    with pytest.raises(ValueError):
        ConfidenceScore(1.5, source="fake")


# Um SemanticClaim sem nenhuma Evidence não é auditável (não há como rastrear de
# onde veio o claim) e deve ser rejeitado na construção.
def test_claim_requires_at_least_one_evidence() -> None:
    with pytest.raises(ValueError):
        SemanticClaim(ClaimKind.LABEL, "box", ConfidenceScore(0.9, source="fake"), (), _provenance())


# Um único claim nunca pode contradizer a si mesmo — contradicting_claims deve
# retornar vazio.
def test_single_claim_has_no_contradiction() -> None:
    claims = (_claim(ClaimKind.LABEL, "box"),)
    assert contradicting_claims(claims, ClaimKind.LABEL) == ()


# Múltiplos claims do mesmo kind com o mesmo value (concordantes) não devem ser
# reportados como contradição, mesmo com confidences diferentes.
def test_multiple_agreeing_claims_have_no_contradiction() -> None:
    claims = (_claim(ClaimKind.LABEL, "box"), _claim(ClaimKind.LABEL, "box", 0.5))
    assert contradicting_claims(claims, ClaimKind.LABEL) == ()


# Confirma o design "claims-not-labels" (ver research-traceability.md): hipóteses
# conflitantes do mesmo kind coexistem no resultado e são detectáveis via
# contradicting_claims, em vez de o sistema forçar uma única resposta.
def test_contradictory_hypotheses_coexist_and_are_detected() -> None:
    claims = (_claim(ClaimKind.LABEL, "box"), _claim(ClaimKind.LABEL, "crate"))
    contradictions = contradicting_claims(claims, ClaimKind.LABEL)
    assert len(contradictions) == 2
    assert {claim.value for claim in contradictions} == {"box", "crate"}
