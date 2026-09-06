"""Testes de contract de claims semânticos, confidence, evidence e provenance (#156)."""

from __future__ import annotations

import pytest

from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.semantics import (
    ClaimKind,
    ConfidenceScore,
    Evidence,
    RegionKind,
    SemanticClaim,
    contradicting_claims,
    most_confident_claim,
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


# RegionKind separa o que a região *é* (objeto contável, superfície, parte) do
# label livre que o VLM escreveu. UNKNOWN existe para que "o modelo não disse"
# nunca seja confundido com "o modelo disse thing".
def test_region_kind_covers_thing_stuff_part_and_unknown() -> None:
    assert {kind.value for kind in RegionKind} == {"thing", "stuff", "part", "unknown"}


# O valor serializado de RegionKind é a string minúscula do próprio contrato do
# VLM, para que parsing e serialização não precisem de tabela de tradução.
def test_region_kind_value_matches_the_vlm_contract_string() -> None:
    assert RegionKind("thing") is RegionKind.THING
    assert RegionKind.UNKNOWN.value == "unknown"


# Caso central do passo: um claim pode legitimamente não ter score. None significa
# "o produtor não forneceu", e é diferente de 0.0 (score baixo informado).
def test_claim_accepts_absent_confidence() -> None:
    claim = SemanticClaim(ClaimKind.LABEL, "floor", None, (Evidence("e"),), _provenance())
    assert claim.confidence is None


# Um score de 0.0 continua sendo um score informado, não ausência — a distinção
# que o passo inteiro existe para preservar.
def test_zero_confidence_is_a_score_not_an_absence() -> None:
    claim = SemanticClaim(
        ClaimKind.LABEL, "floor", ConfidenceScore(0.0, source="fake"), (Evidence("e"),), _provenance()
    )
    assert claim.confidence is not None
    assert claim.confidence.value == 0.0


# Constrói um claim de label sem score; a hipótese que o VLM devolveu mas não pontuou.
def _unscored(value: str) -> SemanticClaim:
    return SemanticClaim(ClaimKind.LABEL, value, None, (Evidence("e"),), _provenance())


# Caso central da política §6: uma claim pontuada sempre vence uma não pontuada.
def test_scored_claim_outranks_unscored() -> None:
    winner = most_confident_claim((_unscored("carpet"), _claim(ClaimKind.LABEL, "floor", 0.62)))
    assert winner is not None
    assert winner.value == "floor"


# Impede o "conserto" futuro de tratar None como um default alto: mesmo com score
# ridiculamente baixo, a pontuada continua vencendo a não pontuada.
def test_unscored_never_wins_over_scored_even_at_low_confidence() -> None:
    winner = most_confident_claim((_unscored("carpet"), _claim(ClaimKind.LABEL, "floor", 0.01)))
    assert winner is not None
    assert winner.value == "floor"


# Entre claims pontuadas, vence a de maior confiança — o comportamento que já existia.
def test_most_confident_scored_claim_wins() -> None:
    claims = (_claim(ClaimKind.LABEL, "box", 0.4), _claim(ClaimKind.LABEL, "crate", 0.9))
    winner = most_confident_claim(claims)
    assert winner is not None
    assert winner.value == "crate"


# Quando nada foi pontuado não há decisão a tomar. Devolver a primeira seria uma
# escolha arbitrária disfarçada de resultado.
def test_all_unscored_claims_yield_no_decision() -> None:
    assert most_confident_claim((_unscored("carpet"), _unscored("floor"))) is None


# Sem claims não há decisão.
def test_no_claims_yield_no_decision() -> None:
    assert most_confident_claim(()) is None
