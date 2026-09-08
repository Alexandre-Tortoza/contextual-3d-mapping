"""Testes do contract estruturado de hipóteses de label (#202).

A execução real de ``corridor-02-002`` mostrou três perdas estruturais entre a
resposta do VLM e a ``VisualObservation`` persistida: ``category`` e ``kind``
eram descartados, primary e alternative ficavam indistinguíveis, e o modelo
podia repetir o primary dentro de ``alternatives`` gerando claims equivalentes
(``carpet 0.9 / carpet 0.9 / carpet 0.1``). Estes testes fixam o contract que
fecha as três.
"""

from __future__ import annotations

import pytest

from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.semantics import (
    ClaimKind,
    ConfidenceScore,
    Evidence,
    HypothesisRole,
    RegionKind,
    SemanticClaim,
)


# Provenance mínima reutilizável, para que os testes falem só sobre o contract
# de hipótese e não sobre detalhes de proveniência.
def _provenance() -> ModelProvenance:
    return ModelProvenance(stage="test", producer="fake", config_fingerprint="abc123")


# Constrói um claim LABEL válido com papel explícito — o bloco básico dos casos
# abaixo.
def _label(
    value: str,
    *,
    role: HypothesisRole = HypothesisRole.PRIMARY,
    confidence: float | None = None,
    category: str | None = None,
    region_kind: RegionKind | None = None,
) -> SemanticClaim:
    return SemanticClaim(
        ClaimKind.LABEL,
        value,
        None if confidence is None else ConfidenceScore(confidence, source="fake"),
        (Evidence("evidence"),),
        _provenance(),
        role=role,
        category=category,
        region_kind=region_kind,
    )


# O papel da hipótese precisa ser explícito: recuperar "quem é o primary" por
# posição na tupla foi o que deixou diagnostics e merge discordando.
def test_label_claim_requires_an_explicit_role() -> None:
    with pytest.raises(ValueError, match="role"):
        SemanticClaim(
            ClaimKind.LABEL, "wall", None, (Evidence("evidence"),), _provenance()
        )


# ``category`` e ``region_kind`` descrevem a interpretação de identidade. Num
# claim que não é de identidade eles não teriam significado, e aceitá-los
# silenciosamente esconderia um erro do produtor.
@pytest.mark.parametrize(
    ("field", "value"),
    [("role", HypothesisRole.PRIMARY), ("category", "wall"), ("region_kind", RegionKind.STUFF)],
)
def test_identity_fields_are_rejected_on_non_label_claims(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        SemanticClaim(
            ClaimKind.ATTRIBUTE,
            "smooth",
            None,
            (Evidence("evidence"),),
            _provenance(),
            **{field: value},
        )


# ``category`` é uma categoria semântica mais estável que o label aberto, e os
# dois precisam sobreviver separados: "plain wall" (label) e "wall" (category)
# não são a mesma informação.
def test_category_is_preserved_separately_from_the_label() -> None:
    claim = _label("plain wall", category="wall", region_kind=RegionKind.STUFF)
    assert claim.value == "plain wall"
    assert claim.category == "wall"
    assert claim.region_kind is RegionKind.STUFF


# Uma category vazia é ausência disfarçada de valor. Ausência é ``None``.
def test_empty_category_is_rejected() -> None:
    with pytest.raises(ValueError, match="category"):
        _label("wall", category="")


# Ausência de confidence continua sendo ``None``, nunca um score âncora.
# O run atual reporta 0.9 em 60/60 regiões; se o modelo omitir, o contract
# precisa registrar a omissão em vez de inventar um número.
def test_missing_confidence_stays_none_on_a_hypothesis() -> None:
    assert _label("wall").confidence is None


# Confidence informada é preservada exatamente como veio do produtor.
def test_reported_confidence_is_preserved_verbatim() -> None:
    claim = _label("wall", confidence=0.9)
    assert claim.confidence is not None
    assert claim.confidence.value == pytest.approx(0.9)
    assert claim.confidence.source == "fake"


# A confiança reportada pelo produtor nunca é a confiança calibrada: o campo
# calibrado vive em ``support`` e permanece ausente até existir calibração real.
def test_reported_confidence_is_not_a_calibrated_confidence() -> None:
    claim = _label("wall", confidence=0.9)
    assert claim.support is None
