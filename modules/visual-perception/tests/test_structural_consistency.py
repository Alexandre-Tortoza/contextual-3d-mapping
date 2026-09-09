"""Testes da política de coerência entre conceito e natureza de região (#216)."""

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
from visual_perception.domain.structural_consistency import (
    StructuralVerdict,
    expected_region_kind,
    head_noun,
    region_kind_verdict,
    singularize,
)

_PROVENANCE = ModelProvenance(stage="region_semantics", producer="fake", config_fingerprint="abc")


# Constrói uma claim de identidade com o conceito, a categoria e a natureza
# pedidos, que são exatamente os três campos que a política inspeciona.
def _claim(
    value: str, kind: RegionKind | None = RegionKind.THING, category: str | None = None
) -> SemanticClaim:
    return SemanticClaim(
        ClaimKind.LABEL,
        value,
        ConfidenceScore(0.9, source="fake"),
        (Evidence("e"),),
        _PROVENANCE,
        role=HypothesisRole.PRIMARY,
        category=category,
        region_kind=kind,
    )


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("tiles", "tile"),
        ("trees", "tree"),
        ("crops", "crop"),
        ("flowers", "flower"),
        # Sufixos protegidos: singularizar aqui produziria palavras inexistentes
        # e faria a política casar (ou deixar de casar) por acidente.
        ("glass", "glass"),
        ("grass", "grass"),
        ("chassis", "chassis"),
        ("sky", "sky"),
    ],
)
def test_singularization_is_minimal_and_protects_false_plurals(word: str, expected: str) -> None:
    """A singularização remove plural real e preserva sufixos que não são plural."""
    assert singularize(word) == expected


@pytest.mark.parametrize(
    ("concept", "expected"),
    [
        ("plain wall", "wall"),
        ("indoor ceiling", "ceiling"),
        ("ceiling tiles", "tile"),
        ("wooden floor", "floor"),
        # O caso que justifica casar pelo núcleo, e não por qualquer palavra:
        # uma luminária de teto é um objeto contável, e casar ``ceiling`` em
        # qualquer posição a classificaria como superfície contínua.
        ("ceiling light fixture", "fixture"),
        ("row of crops", "crop"),
    ],
)
def test_the_head_noun_is_the_last_word_of_the_compound(concept: str, expected: str) -> None:
    """O núcleo de um composto nominal em inglês é a última palavra, singularizada."""
    assert head_noun(concept) == expected


# Uma parede é matéria contínua em qualquer cena: a política diz isso, e é a
# única coisa que ela diz.
def test_an_unambiguous_surface_concept_expects_stuff() -> None:
    """Conceitos inequivocamente contínuos exigem ``STUFF``."""
    assert expected_region_kind(_claim("plain wall")) is RegionKind.STUFF
    assert region_kind_verdict(_claim("plain wall", RegionKind.THING)) is StructuralVerdict.CONTRADICTS
    assert region_kind_verdict(_claim("plain wall", RegionKind.STUFF)) is StructuralVerdict.SUPPORTS


# O caso que a lista curta protege: um conceito fora dela é indeterminado, e não
# "correto por omissão". Ampliar a lista até o número ficar bonito mascararia o
# erro do reasoner em vez de expô-lo.
@pytest.mark.parametrize("concept", ["wooden panel", "car bumper", "building interior", "tree"])
def test_an_ambiguous_concept_stays_undetermined(concept: str) -> None:
    """Conceitos fora do conjunto inequívoco não recebem veredito."""
    assert expected_region_kind(_claim(concept)) is None
    assert region_kind_verdict(_claim(concept, RegionKind.THING)) is StructuralVerdict.UNDETERMINED


# O contra-exemplo que motivou casar pelo núcleo: a palavra ``ceiling`` aparece,
# mas o sujeito é a luminária.
def test_a_fixture_named_after_a_surface_is_not_a_surface() -> None:
    """``ceiling light fixture`` não é classificado como superfície contínua."""
    assert expected_region_kind(_claim("ceiling light fixture")) is None


# A categoria é consultada só quando o label não decide: ela é uma
# generalização, e uma generalização não prevalece sobre a afirmação específica.
def test_the_category_is_consulted_only_when_the_label_is_undetermined() -> None:
    """Uma categoria inequívoca decide quando o label é ambíguo."""
    assert expected_region_kind(_claim("wooden panel", category="wall")) is RegionKind.STUFF
    # E não o contrário: o label específico continua mandando.
    assert expected_region_kind(_claim("plain wall", category="furniture")) is RegionKind.STUFF


# Sem natureza declarada não há o que contradizer: ``unknown`` é ausência, e
# ausência nunca vira erro.
@pytest.mark.parametrize("kind", [None, RegionKind.UNKNOWN])
def test_an_undeclared_kind_is_never_a_contradiction(kind: RegionKind | None) -> None:
    """Uma natureza ausente ou desconhecida não produz contradição."""
    assert region_kind_verdict(_claim("plain wall", kind)) is StructuralVerdict.UNDETERMINED
