"""Testes da política de evidência contextual publicável."""

from __future__ import annotations

import numpy as np
import pytest

from visual_perception.domain.contextual_evidence import (
    GENERIC_STRUCTURAL_HEAD_NOUNS,
    ContextualEvidenceVerdict,
    adjectival_stem,
    contextual_evidence_verdict,
    host_surface_concept,
    material_roots,
)
from visual_perception.domain.geometry import Mask
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.semantics import (
    ClaimKind,
    ConfidenceScore,
    Evidence,
    HypothesisRole,
    RegionKind,
    SemanticClaim,
)

_PROVENANCE = ModelProvenance(stage="region_semantics", producer="fake", config_fingerprint="abc")


# Constrói uma hipótese de identidade afirmada com o conceito pedido.
def _label(
    value: str,
    *,
    role: HypothesisRole = HypothesisRole.PRIMARY,
    category: str | None = None,
) -> SemanticClaim:
    return SemanticClaim(
        ClaimKind.LABEL,
        value,
        ConfidenceScore(0.9, source="fake"),
        (Evidence("raw region response"),),
        _PROVENANCE,
        role=role,
        category=category,
        region_kind=RegionKind.THING,
    )


# Constrói uma claim descritiva de um kind não pontuado, como o produtor a
# devolve junto do label.
def _descriptive(kind: ClaimKind, value: str) -> SemanticClaim:
    return SemanticClaim(kind, value, None, (Evidence("raw region response"),), _PROVENANCE)


# Constrói uma região mínima com as claims pedidas: a política não olha
# geometria, então a máscara só precisa ser válida.
def _region(*claims: SemanticClaim) -> ObservedRegion:
    data = np.zeros((8, 8), dtype=np.bool_)
    data[2:6, 2:6] = True
    mask = Mask(data, 8, 8)
    return ObservedRegion("region-1", mask, mask.bounding_box(), 0.9, ("p-1",), claims=claims)


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("wooden", "wood"),
        ("metallic", "metal"),
        ("reddish", "red"),
        ("rusty", "rust"),
        # Palavras curtas demais para reduzir: o radical viraria um fragmento
        # que casaria por acaso com qualquer outro material.
        ("wood", "wood"),
        ("icy", "icy"),
        # Redução aproximada: ``plastic`` apenas termina como adjetivo. É
        # inofensivo porque a comparação literal com o material declarado vem
        # antes, e uma palavra que já bate nunca chega ao radical.
        ("plastic", "plast"),
    ],
)
def test_the_adjectival_stem_links_a_modifier_to_its_material(word: str, expected: str) -> None:
    """O radical adjetival liga ``wooden`` a ``wood`` sem virar um lematizador."""
    assert adjectival_stem(word) == expected


@pytest.mark.parametrize(
    "concept",
    ["wall", "floor", "ceiling", "walls", "plain wall", "blank ceiling", "ceiling tiles"],
)
def test_a_generic_structural_surface_is_not_contextual_evidence(concept: str) -> None:
    """Uma superfície estrutural genérica não conta como evidência contextual."""
    verdict = contextual_evidence_verdict(_region(_label(concept)))
    assert verdict is ContextualEvidenceVerdict.GENERIC_STRUCTURAL_SURFACE


@pytest.mark.parametrize(
    ("concept", "material"),
    [("wooden floor", "wood"), ("wooden panel", "wood"), ("metallic ceiling", "metal")],
)
def test_a_modifier_the_region_already_filed_as_material_adds_no_evidence(
    concept: str, material: str
) -> None:
    """``wooden floor`` com ``material: wood`` continua sendo piso genérico."""
    region = _region(_label(concept), _descriptive(ClaimKind.MATERIAL, material))
    assert contextual_evidence_verdict(region) is ContextualEvidenceVerdict.GENERIC_STRUCTURAL_SURFACE


@pytest.mark.parametrize(
    "concept",
    [
        "cracked wall",
        "graffiti on wall",
        "warning text on wall",
        "peeled paint",
        "structural crack",
        "water stain",
        "door",
        "ceiling light fixture",
        "exposed reinforcement",
    ],
)
def test_evidence_over_a_surface_is_published(concept: str) -> None:
    """Dano, texto e objeto sobre a superfície continuam sendo evidência."""
    assert contextual_evidence_verdict(_region(_label(concept))) is ContextualEvidenceVerdict.CONTEXT_BEARING


# O canal ``condition`` é integridade, e integridade é exatamente o que a
# política existe para preservar. Descontá-lo como se fosse aparência engoliria
# a evidência — é por isso que o desconto olha só ``material``.
def test_a_condition_never_discounts_the_evidence_it_describes() -> None:
    """``cracked wall`` com ``condition: cracked`` continua publicável."""
    region = _region(_label("cracked wall"), _descriptive(ClaimKind.CONDITION, "cracked"))
    assert contextual_evidence_verdict(region) is ContextualEvidenceVerdict.CONTEXT_BEARING


def test_an_attribute_never_discounts_the_evidence_it_describes() -> None:
    """Um atributo não silencia o modificador: só ``material`` desconta."""
    region = _region(_label("cracked wall"), _descriptive(ClaimKind.ATTRIBUTE, "cracked"))
    assert contextual_evidence_verdict(region) is ContextualEvidenceVerdict.CONTEXT_BEARING


# Esta é a regra que impede a política de virar uma substituição de label. Uma
# região que o produtor não decidiu entre parede e porta sai do output público;
# ela nunca é promovida a porta porque parede deixou de ser publicável.
def test_an_alternative_hypothesis_never_makes_a_surface_publishable() -> None:
    """A dúvida registrada pelo produtor não promove a região a evidência."""
    region = _region(
        _label("wall"),
        _label("door", role=HypothesisRole.ALTERNATIVE),
    )
    assert contextual_evidence_verdict(region) is ContextualEvidenceVerdict.GENERIC_STRUCTURAL_SURFACE


# Um passe de refinamento acrescenta uma segunda claim PRIMARY. Julgar só a
# primeira suprimiria uma região que o refinamento já tinha reinterpretado.
def test_a_refined_reinterpretation_is_enough_to_publish() -> None:
    """Basta uma hipótese afirmada não estrutural para a região ser publicada."""
    region = _region(_label("wall"), _label("fire extinguisher"))
    assert contextual_evidence_verdict(region) is ContextualEvidenceVerdict.CONTEXT_BEARING


def test_a_region_without_asserted_identity_is_uninterpreted() -> None:
    """Sem identidade afirmada não há base para julgar, e o desfecho é nomeado."""
    assert contextual_evidence_verdict(_region()) is ContextualEvidenceVerdict.UNINTERPRETED


# ``category`` é uma generalização, e deixar a generalização suprimir a
# afirmação específica inverteria o contract: uma porta continua sendo porta.
def test_the_category_never_suppresses_the_specific_assertion() -> None:
    """Um label ``door`` com categoria ``wall`` continua sendo evidência."""
    region = _region(_label("door", category="wall"))
    assert contextual_evidence_verdict(region) is ContextualEvidenceVerdict.CONTEXT_BEARING


def test_the_composition_can_declare_extra_structural_head_nouns() -> None:
    """Núcleos extras da configuração entram na política sem recompilar o domínio."""
    region = _region(_label("scaffolding"))
    assert contextual_evidence_verdict(region) is ContextualEvidenceVerdict.CONTEXT_BEARING
    assert (
        contextual_evidence_verdict(region, extra_head_nouns=frozenset({"scaffolding"}))
        is ContextualEvidenceVerdict.GENERIC_STRUCTURAL_SURFACE
    )


@pytest.mark.parametrize(
    ("concept", "expected"),
    [
        ("cracked wall", "wall"),
        ("graffiti on wall", "wall"),
        ("wall graffiti", "wall"),
        ("water stain on ceiling", "ceiling"),
        # Nenhuma superfície citada: host desconhecido é um desfecho legítimo.
        ("structural crack", None),
        ("door", None),
        # Duas superfícies citadas: afirmar uma delas seria forçar.
        ("wall and ceiling junction", None),
    ],
)
def test_the_host_surface_comes_only_from_the_asserted_identity(
    concept: str, expected: str | None
) -> None:
    """A superfície hospedeira é a que a própria identidade nomeia, ou nenhuma."""
    assert host_surface_concept(_region(_label(concept))) == expected


def test_disagreeing_hypotheses_do_not_produce_a_host_surface() -> None:
    """Hipóteses afirmadas que discordam deixam o host desconhecido."""
    region = _region(_label("cracked wall"), _label("fire extinguisher"))
    assert host_surface_concept(region) is None


def test_the_material_roots_come_from_the_regions_own_material_claims() -> None:
    """Os radicais de material saem do que a própria região declarou."""
    region = _region(
        _label("wooden panel"),
        _descriptive(ClaimKind.MATERIAL, "painted wood"),
        _descriptive(ClaimKind.ATTRIBUTE, "smooth"),
    )
    assert material_roots(region) == frozenset({"painted", "wood"})


# O conjunto do audit precisa continuar minúsculo, porque ampliá-lo mascararia
# o erro do reasoner. O da publicação responde outra pergunta e pode crescer.
def test_the_publication_set_extends_the_audit_invariant_without_replacing_it() -> None:
    """A política de publicação parte do invariante do audit e o estende."""
    from visual_perception.domain.structural_consistency import INHERENTLY_STUFF_HEAD_NOUNS

    assert INHERENTLY_STUFF_HEAD_NOUNS < GENERIC_STRUCTURAL_HEAD_NOUNS
    assert "panel" in GENERIC_STRUCTURAL_HEAD_NOUNS
    assert "panel" not in INHERENTLY_STUFF_HEAD_NOUNS


# Uma região publicada por causa da segunda hipótese precisa se apresentar por
# ela. Sem isto, o overlay voltaria a exibir o label estrutural que a política
# existe para tirar da frente.
def test_the_evidence_bearing_claim_is_the_one_that_made_it_publishable() -> None:
    """A claim portadora é a primeira afirmada que não é superfície genérica."""
    from visual_perception.domain.contextual_evidence import contextual_evidence_claim

    region = _region(_label("wall"), _label("red wall"))
    claim = contextual_evidence_claim(region)

    assert claim is not None
    assert claim.value == "red wall"


def test_a_structural_surface_has_no_evidence_bearing_claim() -> None:
    """Sem hipótese portadora, quem apresenta a região cai para o primário."""
    from visual_perception.domain.contextual_evidence import contextual_evidence_claim

    assert contextual_evidence_claim(_region(_label("plain wall"))) is None
