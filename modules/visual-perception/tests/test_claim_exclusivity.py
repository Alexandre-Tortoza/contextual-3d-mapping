"""Testes da política de exclusividade entre claims semânticos (#202).

O run real de ``corridor-02-002`` mostrou a política antiga marcando como
contraditório qualquer par de claims do mesmo ``ClaimKind`` com texto
diferente. Com isso ``smooth`` contradizia a ``description`` da mesma região, e
toda alternativa legítima virava warning de auditoria.

Contradição depende de o slot ser mutuamente exclusivo, não de os valores serem
diferentes: uma parede pode ser branca **e** lisa **e** danificada, e um objeto
composto pode ter mais de um material.
"""

from __future__ import annotations

import pytest

from visual_perception.domain.claim_exclusivity import claims_compete, competing_claims
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.semantics import (
    IDENTITY_CLAIM_KINDS,
    ClaimKind,
    ConfidenceScore,
    Evidence,
    HypothesisRole,
    SemanticClaim,
)


# Constrói um claim mínimo do kind pedido, declarando papel só quando o kind é
# de identidade — a mesma regra que o contract impõe.
def _claim(kind: ClaimKind, value: str, *, confidence: float | None = None) -> SemanticClaim:
    return SemanticClaim(
        kind,
        value,
        None if confidence is None else ConfidenceScore(confidence, source="fake"),
        (Evidence("e"),),
        ModelProvenance(stage="t", producer="fake", config_fingerprint="abc"),
        role=HypothesisRole.PRIMARY if kind in IDENTITY_CLAIM_KINDS else None,
    )


# Atributos descrevem propriedades que coexistem. "white" e "smooth" na mesma
# parede não são hipóteses concorrentes, e contá-los como contradição inflava
# contradiction_support em todas as 60 regiões do frame.
def test_two_different_attributes_do_not_compete() -> None:
    assert not claims_compete(
        _claim(ClaimKind.ATTRIBUTE, "white"), _claim(ClaimKind.ATTRIBUTE, "smooth")
    )


# Três atributos continuam coexistindo: não há limite implícito de um por slot.
@pytest.mark.parametrize("other", ["white", "smooth", "damaged"])
def test_attributes_never_compete_regardless_of_value(other: str) -> None:
    assert not claims_compete(
        _claim(ClaimKind.ATTRIBUTE, "damaged"), _claim(ClaimKind.ATTRIBUTE, other)
    )


# A descrição livre e um atributo descrevem o mesmo sujeito por ângulos
# diferentes. Eles não se excluem.
def test_description_and_attribute_do_not_compete() -> None:
    assert not claims_compete(
        _claim(ClaimKind.SCENE_DESCRIPTION, "A narrow corridor."),
        _claim(ClaimKind.ATTRIBUTE, "narrow"),
    )


# Um objeto composto tem mais de um material: uma porta pode ser de madeira e
# de vidro. Material diferente não é automaticamente contradição.
def test_different_materials_do_not_compete() -> None:
    assert not claims_compete(_claim(ClaimKind.MATERIAL, "wood"), _claim(ClaimKind.MATERIAL, "glass"))


# Identidade é o slot mutuamente exclusivo por definição: a região não é uma
# parede *e* uma porta. Hipóteses concorrentes continuam competindo.
def test_competing_identity_hypotheses_compete() -> None:
    assert claims_compete(_claim(ClaimKind.LABEL, "wall"), _claim(ClaimKind.LABEL, "door"))


# Duas hipóteses de identidade com o mesmo valor concordam, não competem.
def test_identical_identity_hypotheses_do_not_compete() -> None:
    assert not claims_compete(_claim(ClaimKind.LABEL, "wall"), _claim(ClaimKind.LABEL, "wall"))


# A cena tem um tipo só.
def test_competing_scene_types_compete() -> None:
    assert claims_compete(
        _claim(ClaimKind.SCENE_TYPE, "corridor"), _claim(ClaimKind.SCENE_TYPE, "field")
    )


# Estados explicitamente declarados como incompatíveis competem: uma porta não
# está aberta e fechada ao mesmo tempo.
@pytest.mark.parametrize(("left", "right"), [("open", "closed"), ("on", "off"), ("wet", "dry")])
def test_explicitly_incompatible_states_compete(left: str, right: str) -> None:
    assert claims_compete(_claim(ClaimKind.CONDITION, left), _claim(ClaimKind.CONDITION, right))


# Condições fora dos grupos declarados coexistem. O módulo não tem ontologia de
# estados, e inventar uma tornaria toda condição nova uma contradição.
def test_conditions_outside_a_declared_group_do_not_compete() -> None:
    assert not claims_compete(
        _claim(ClaimKind.CONDITION, "clean"), _claim(ClaimKind.CONDITION, "worn")
    )


# Kinds diferentes descrevem eixos diferentes e nunca competem entre si.
def test_claims_of_different_kinds_never_compete() -> None:
    assert not claims_compete(_claim(ClaimKind.LABEL, "wall"), _claim(ClaimKind.MATERIAL, "wall"))


# Um claim não contradiz a si mesmo.
def test_a_claim_does_not_compete_with_itself() -> None:
    claim = _claim(ClaimKind.LABEL, "wall")
    assert not claims_compete(claim, claim)


# O caso completo da região real: o primary compete só com a alternativa de
# identidade, e ignora atributo, condição e material da mesma região.
def test_only_the_identity_alternative_competes_within_a_real_region() -> None:
    primary = _claim(ClaimKind.LABEL, "plain wall", confidence=0.9)
    claims = (
        primary,
        _claim(ClaimKind.LABEL, "plain floor", confidence=0.1),
        _claim(ClaimKind.ATTRIBUTE, "The subject region appears to be a plain wall."),
        _claim(ClaimKind.ATTRIBUTE, "smooth"),
        _claim(ClaimKind.CONDITION, "clean"),
        _claim(ClaimKind.MATERIAL, "plaster"),
    )
    assert [claim.value for claim in competing_claims(primary, claims)] == ["plain floor"]
