"""Testes da partição entre evidência publicada e contexto estrutural."""

from __future__ import annotations

import numpy as np

from fixtures import image_observation
from visual_perception.application.contextual_publication import (
    PRODUCER,
    partition_observation,
)
from visual_perception.config import ContextualPublicationConfig
from visual_perception.domain.contextual_evidence import RegionSuppressionReason
from visual_perception.domain.geometry import Mask
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.relations import CandidateRelation, RelationSource
from visual_perception.domain.semantics import (
    ClaimKind,
    ConfidenceScore,
    Evidence,
    HypothesisRole,
    RegionKind,
    SemanticClaim,
)
from visual_perception.domain.visual_observation import SceneContext, VisualObservation

_PROVENANCE = ModelProvenance(stage="region_semantics", producer="qwen_vl", config_fingerprint="abc")


# Constrói uma hipótese de identidade afirmada.
def _label(value: str, *, role: HypothesisRole = HypothesisRole.PRIMARY) -> SemanticClaim:
    return SemanticClaim(
        ClaimKind.LABEL,
        value,
        ConfidenceScore(0.9, source="qwen_vl"),
        (Evidence("raw region response"),),
        _PROVENANCE,
        role=role,
        region_kind=RegionKind.THING,
    )


# Constrói uma claim descritiva não pontuada.
def _descriptive(kind: ClaimKind, value: str) -> SemanticClaim:
    return SemanticClaim(kind, value, None, (Evidence("raw region response"),), _PROVENANCE)


# Constrói uma região retangular na faixa horizontal pedida, para que o teste
# controle geometria e identidade separadamente.
def _region(region_id: str, x_min: int, x_max: int, *claims: SemanticClaim) -> ObservedRegion:
    data = np.zeros((32, 96), dtype=np.bool_)
    data[4:20, x_min:x_max] = True
    mask = Mask(data, 96, 32)
    return ObservedRegion(region_id, mask, mask.bounding_box(), 0.9, (f"{region_id}-p",), claims=claims)


# Monta a observação canônica a partir de uma partição já feita, do jeito que o
# pipeline a monta.
def _observation(
    published: tuple[ObservedRegion, ...],
    structural: tuple[ObservedRegion, ...] = (),
    relations: tuple[CandidateRelation, ...] = (),
) -> VisualObservation:
    image = image_observation(width=96, height=32)
    return VisualObservation(
        source=image.source,
        image_width=96,
        image_height=32,
        scene_context=SceneContext(),
        regions=published,
        relations=relations,
        structural_context=structural,
    )


def _default() -> ContextualPublicationConfig:
    return ContextualPublicationConfig()


def test_a_generic_structural_surface_leaves_the_published_output() -> None:
    """Parede, piso e teto isolados não aparecem entre as regiões publicadas."""
    regions = (
        _region("region-wall", 0, 20, _label("wall")),
        _region("region-floor", 20, 40, _label("floor")),
        _region("region-ceiling", 40, 60, _label("ceiling")),
    )
    result = partition_observation(regions, _default())

    assert result.published == ()
    assert {region.region_id for region in result.structural_context} == {
        "region-wall",
        "region-floor",
        "region-ceiling",
    }


def test_a_suppressed_surface_is_preserved_whole_as_internal_context() -> None:
    """A superfície suprimida mantém geometria, claims e alcance por relação."""
    wall = _region("region-wall", 0, 40, _label("wall"))
    crack = _region("region-crack", 10, 20, _label("structural crack"))
    result = partition_observation((crack, wall), _default())

    relation = CandidateRelation(
        relation_id="relation-1",
        subject_region_id="region-crack",
        object_region_id="region-wall",
        predicate="part_of",
        confidence=None,
        evidence=(Evidence("measured containment"),),
        provenance=_PROVENANCE,
        source=RelationSource.MODEL_INFERRED,
    )
    observation = _observation(result.published, result.structural_context, (relation,))

    # A relação atravessa a partição sem virar referência pendurada, e a
    # superfície continua recuperável pelo id.
    assert observation.region_by_id("region-wall").claims == wall.claims
    assert observation.region_by_id("region-wall").mask.area() == wall.mask.area()
    assert len(observation.all_regions) == 2


def test_evidence_on_a_wall_keeps_the_producers_concept_and_gains_a_host() -> None:
    """``cracked wall`` continua ``cracked wall``, e ganha ``host_surface: wall``."""
    result = partition_observation((_region("region-1", 0, 20, _label("cracked wall")),), _default())

    (published,) = result.published
    labels = [claim.value for claim in published.claims if claim.kind is ClaimKind.LABEL]
    hosts = [claim for claim in published.claims if claim.kind is ClaimKind.HOST_SURFACE]

    assert labels == ["cracked wall"]
    assert [claim.value for claim in hosts] == ["wall"]
    assert hosts[0].provenance.producer == PRODUCER
    assert hosts[0].confidence is None


def test_graffiti_on_a_wall_stays_evidence() -> None:
    """Graffiti sobre parede é evidência, com a parede preservada como host."""
    result = partition_observation((_region("region-1", 0, 20, _label("graffiti on wall")),), _default())

    (published,) = result.published
    assert [claim.value for claim in published.claims if claim.kind is ClaimKind.HOST_SURFACE] == ["wall"]


def test_peeled_paint_survives_even_though_it_belongs_to_a_wall() -> None:
    """``peeled paint`` não desaparece por estar sobre uma superfície."""
    result = partition_observation((_region("region-1", 0, 20, _label("peeled paint")),), _default())

    assert len(result.published) == 1
    assert result.structural_context == ()
    # Sem superfície nomeada na identidade, o host fica desconhecido — e
    # ausência é a forma de dizer isso, nunca um valor de preenchimento.
    assert not [claim for claim in result.published[0].claims if claim.kind is ClaimKind.HOST_SURFACE]


# A política reduz falso positivo; ela não troca de classe. Uma região que o
# produtor não decidiu entre parede e porta sai do output público, e nenhum
# label é reescrito.
def test_an_ambiguous_wall_never_becomes_the_door_it_might_be() -> None:
    """``wall`` com alternativa ``door`` sai do output sem virar ``door``."""
    region = _region(
        "region-1",
        0,
        20,
        _label("wall"),
        _label("door", role=HypothesisRole.ALTERNATIVE),
    )
    result = partition_observation((region,), _default())

    assert result.published == ()
    (suppressed,) = result.structural_context
    assert [claim.value for claim in suppressed.claims if claim.kind is ClaimKind.LABEL] == [
        "wall",
        "door",
    ]


def test_a_scene_without_contextual_evidence_publishes_nothing() -> None:
    """Uma cena inteiramente normal pode não publicar nenhuma região."""
    regions = tuple(
        _region(f"region-{index}", index * 10, index * 10 + 8, _label("plain wall"))
        for index in range(4)
    )
    result = partition_observation(regions, _default())
    observation = _observation(result.published, result.structural_context)

    assert observation.regions == ()
    assert len(observation.structural_context) == 4


def test_every_suppression_records_its_reason_and_concept() -> None:
    """Cada supressão registra motivo e conceito, como a filtragem de proposals."""
    regions = (
        _region("region-1", 0, 20, _label("plain wall")),
        _region("region-2", 20, 40, _label("door")),
    )
    result = partition_observation(regions, _default())

    (record,) = result.records
    assert record.region_id == "region-1"
    assert record.reason is RegionSuppressionReason.GENERIC_STRUCTURAL_SURFACE
    assert record.concept == "plain wall"


def test_a_wooden_floor_is_still_a_floor() -> None:
    """Um modificador que repete o material declarado não publica a superfície."""
    region = _region(
        "region-1", 0, 20, _label("wooden floor"), _descriptive(ClaimKind.MATERIAL, "wood")
    )
    assert partition_observation((region,), _default()).published == ()


# Desligar a política precisa devolver exatamente o comportamento anterior a
# ela: é o que permite a uma comparação atribuir um efeito à política sozinha.
def test_the_disabled_policy_publishes_everything_untouched() -> None:
    """Desligada, a partição não suprime nada nem anexa claim nenhuma."""
    regions = (
        _region("region-1", 0, 20, _label("wall")),
        _region("region-2", 20, 40, _label("cracked wall")),
    )
    result = partition_observation(regions, ContextualPublicationConfig(enabled=False))

    assert result.published == regions
    assert result.structural_context == ()
    assert result.records == ()


def test_the_partition_never_touches_geometry() -> None:
    """Nenhuma máscara, box ou id muda ao atravessar a partição."""
    regions = (
        _region("region-1", 0, 20, _label("wall")),
        _region("region-2", 20, 40, _label("fire extinguisher")),
    )
    result = partition_observation(regions, _default())

    for original, kept in zip(regions, (result.structural_context[0], result.published[0]), strict=True):
        assert kept.region_id == original.region_id
        assert kept.box == original.box
        assert np.array_equal(kept.mask.data, original.mask.data)


def test_the_extra_head_nouns_from_config_reach_the_policy() -> None:
    """O vocabulário extra da composição chega ao domínio pela configuração."""
    region = _region("region-1", 0, 20, _label("wall scaffolding"))
    config = ContextualPublicationConfig(extra_structural_head_nouns=("scaffolding",))

    assert partition_observation((region,), _default()).published != ()
    assert partition_observation((region,), config).published == ()


# O vocabulário extra amplia o que conta como superfície, e não o que conta
# como evidência: um dano sobre a superfície declarada continua publicável.
def test_the_extra_head_nouns_do_not_swallow_evidence_on_that_surface() -> None:
    """``cracked wall scaffolding`` continua publicável com o vocabulário extra."""
    region = _region("region-1", 0, 20, _label("cracked wall scaffolding"))
    config = ContextualPublicationConfig(extra_structural_head_nouns=("scaffolding",))

    assert partition_observation((region,), config).published != ()
