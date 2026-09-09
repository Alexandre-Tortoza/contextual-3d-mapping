"""Testes da reconciliação contextual intra-frame (#205)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from visual_perception.application.reconciliation import (
    PRODUCER,
    canonical_concept,
    reconcile_observation,
    reconciled_label_claim,
    region_concept,
)
from visual_perception.config import ReconciliationConfig
from visual_perception.domain.contextual_entities import EntityHypothesisKind, EntityHypothesisStatus
from visual_perception.domain.embeddings import EmbeddingModality, EmbeddingSpace, VisualEmbedding
from visual_perception.domain.geometry import Mask
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.regions import ObservedRegion, primary_label_claim
from visual_perception.domain.semantics import (
    ClaimKind,
    ConfidenceScore,
    Evidence,
    HypothesisRole,
    RegionKind,
    SemanticClaim,
)

_PROVENANCE = ModelProvenance(stage="region_semantics", producer="fake", config_fingerprint="abc")
_VISUAL_SPACE = EmbeddingSpace(
    "dinov2", "facebook/dinov2-base", 4, EmbeddingModality.VISUAL_DENSE
)


# Constrói uma claim de identidade com o conceito e a natureza pedidos.
def _claim(value: str, kind: RegionKind = RegionKind.THING, category: str | None = None) -> SemanticClaim:
    return SemanticClaim(
        ClaimKind.LABEL,
        value,
        ConfidenceScore(0.9, source="fake"),
        (Evidence("raw region response"),),
        _PROVENANCE,
        role=HypothesisRole.PRIMARY,
        category=category,
        region_kind=kind,
    )


# Constrói uma região retangular na faixa horizontal pedida, para que
# adjacência entre regiões seja controlada pelo teste.
def _region(region_id: str, x_min: int, x_max: int, claims: tuple[SemanticClaim, ...]) -> ObservedRegion:
    data = np.zeros((32, 96), dtype=np.bool_)
    data[4:20, x_min:x_max] = True
    mask = Mask(data, 96, 32)
    return ObservedRegion(region_id, mask, mask.bounding_box(), 0.9, (f"{region_id}-p",), claims=claims)


# Constrói um embedding denso unitário na direção pedida, para que a coerência
# de um grupo seja escolhida pelo teste em vez de emergir de um fake.
def _embedding(region_id: str, direction: list[float]) -> VisualEmbedding:
    vector = np.asarray(direction, dtype=np.float64)
    vector = vector / np.linalg.norm(vector)
    return VisualEmbedding(
        embedding_id=f"visual-{region_id}",
        region_id=region_id,
        vector=tuple(vector.tolist()),
        dimension=len(direction),
        pooling_method="pixel_nearest_highres",
        feature_resolution="4x4",
        model_id="dinov2",
        normalized=True,
    )


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("plain wall", "wall"),
        ("Wall", "wall"),
        ("walls", "wall"),
        ("long, narrow hallway", "long narrow hallway"),
        ("trees", "tree"),
        ("ceiling tiles", "ceiling tile"),
        # A canonicalização é lexical e para aí: ``ceiling tile`` e ``ceiling``
        # continuam conceitos distintos, porque afirmar que um é parte do outro
        # é uma afirmação sobre o mundo, e o lugar dela é uma relação.
        ("ceiling", "ceiling"),
    ],
)
def test_canonicalization_is_lexical_and_minimal(label: str, expected: str) -> None:
    """A forma canônica colapsa grafia e plural, e não consulta ontologia."""
    assert canonical_concept(label) == expected


# A garantia central: o label cru sobrevive intacto, e a interpretação
# reconciliada aparece ao lado dele, com produtor próprio.
def test_the_raw_label_is_preserved_and_the_reconciled_claim_is_added() -> None:
    """A reconciliação acrescenta uma claim nova sem tocar na original."""
    region = _region("region-a", 4, 20, (_claim("plain wall", RegionKind.THING),))

    result = reconcile_observation((region,), "frame-1", ReconciliationConfig())

    reconciled = result.regions[0]
    primary = primary_label_claim(reconciled)
    assert primary is not None
    assert primary.value == "plain wall"
    assert primary.region_kind is RegionKind.THING
    assert primary.provenance.producer == "fake"

    derived = reconciled_label_claim(reconciled)
    assert derived is not None
    assert derived.role is HypothesisRole.RECONCILED
    assert derived.value == "wall"
    assert derived.region_kind is RegionKind.STUFF
    assert derived.provenance.producer == PRODUCER
    # A evidência da claim reconciliada nomeia o que a produziu.
    assert "plain wall" in derived.evidence[0].description
    assert "thing" in derived.evidence[0].description


# Sem nada a acrescentar, nada é acrescentado: uma região já canônica e coerente
# não ganha uma cópia de si mesma.
def test_a_region_that_is_already_canonical_gets_no_extra_claim() -> None:
    """Uma região coerente e já canônica não recebe claim reconciliada."""
    region = _region("region-a", 4, 20, (_claim("wall", RegionKind.STUFF),))

    result = reconcile_observation((region,), "frame-1", ReconciliationConfig())

    assert len(result.regions[0].claims) == 1
    assert result.records[0].structural_verdict.value == "supports"


# O caso medido: três recortes contíguos da mesma parede, rotulados de formas
# ligeiramente diferentes, viram um grupo — sem que nenhum deles desapareça.
def test_touching_regions_with_the_same_concept_form_a_group() -> None:
    """Regiões contíguas de mesmo conceito viram uma hipótese de entidade."""
    regions = (
        _region("region-a", 4, 20, (_claim("wall", RegionKind.THING),)),
        _region("region-b", 20, 36, (_claim("plain wall", RegionKind.THING),)),
        _region("region-c", 36, 52, (_claim("walls", RegionKind.THING),)),
    )
    embeddings = tuple(
        _embedding(region.region_id, [1.0, 0.1, 0.0, 0.0]) for region in regions
    )

    result = reconcile_observation(
        regions,
        "frame-1",
        ReconciliationConfig(),
        visual_embeddings=embeddings,
        visual_space=_VISUAL_SPACE,
    )

    assert len(result.entities) == 1
    entity = result.entities[0]
    assert entity.kind is EntityHypothesisKind.SAME_SURFACE
    assert entity.canonical_concept == "wall"
    assert entity.region_kind is RegionKind.STUFF
    assert entity.member_region_ids == ("region-a", "region-b", "region-c")
    assert entity.member_raw_labels == ("wall", "plain wall", "walls")
    assert entity.status is EntityHypothesisStatus.SUPPORTED
    assert entity.feature_coherence == pytest.approx(1.0)
    # Nenhuma região foi removida, e a geometria de cada uma segue intacta.
    assert len(result.regions) == 3
    assert [region.box for region in result.regions] == [region.box for region in regions]
    assert [region.mask.area() for region in result.regions] == [
        region.mask.area() for region in regions
    ]


# A corroboração pode faltar, e o grupo continua existindo: contato e conceito
# ainda são evidência. A medição foi explícita quanto a isso — o cosseno denso
# sozinho separa mesmo-label de label-diferente com acurácia balanceada de 0,66.
def test_a_group_with_low_coherence_stays_unresolved() -> None:
    """Coerência densa abaixo do limiar rebaixa o grupo, sem apagá-lo."""
    regions = (
        _region("region-a", 4, 20, (_claim("wall", RegionKind.STUFF),)),
        _region("region-b", 20, 36, (_claim("wall", RegionKind.STUFF),)),
    )
    embeddings = (
        _embedding("region-a", [1.0, 0.0, 0.0, 0.0]),
        _embedding("region-b", [0.0, 1.0, 0.0, 0.0]),
    )

    result = reconcile_observation(
        regions,
        "frame-1",
        ReconciliationConfig(),
        visual_embeddings=embeddings,
        visual_space=_VISUAL_SPACE,
    )

    assert len(result.entities) == 1
    assert result.entities[0].status is EntityHypothesisStatus.UNRESOLVED
    assert "below the configured" in (result.entities[0].reason or "")


# Sem evidência densa não há corroboração possível, e o grupo declara isso em
# vez de se apresentar como confirmado.
def test_a_group_without_dense_evidence_is_unresolved() -> None:
    """Um grupo sem embeddings não é apresentado como corroborado."""
    regions = (
        _region("region-a", 4, 20, (_claim("wall", RegionKind.STUFF),)),
        _region("region-b", 20, 36, (_claim("wall", RegionKind.STUFF),)),
    )

    result = reconcile_observation(regions, "frame-1", ReconciliationConfig())

    assert result.entities[0].status is EntityHypothesisStatus.UNRESOLVED
    assert result.entities[0].feature_coherence is None
    assert result.entities[0].space is None


# Duas cadeiras encostadas continuam duas cadeiras: um ``thing`` nunca entra em
# um grupo de mesma superfície, porque agrupá-lo inventaria identidade.
def test_countable_objects_are_never_grouped_as_one_surface() -> None:
    """Regiões reconciliadas como ``thing`` não formam grupo de superfície."""
    regions = (
        _region("region-a", 4, 20, (_claim("chair", RegionKind.THING),)),
        _region("region-b", 20, 36, (_claim("chair", RegionKind.THING),)),
    )

    assert reconcile_observation(regions, "frame-1", ReconciliationConfig()).entities == ()


# Regiões distantes não são a mesma superfície, por mais que compartilhem o
# conceito: contato espacial é condição, não detalhe.
def test_regions_that_do_not_touch_are_not_grouped() -> None:
    """Conceito igual sem contato não produz grupo."""
    regions = (
        _region("region-a", 4, 20, (_claim("wall", RegionKind.STUFF),)),
        _region("region-b", 60, 76, (_claim("wall", RegionKind.STUFF),)),
    )

    assert reconcile_observation(regions, "frame-1", ReconciliationConfig()).entities == ()


# O id do grupo é estável e independente da ordem em que as regiões chegam:
# dois runs da mesma configuração precisam ser comparáveis linha a linha.
def test_entity_ids_are_stable_and_order_independent() -> None:
    """O mesmo conjunto de membros produz sempre o mesmo id de entidade."""
    regions = (
        _region("region-a", 4, 20, (_claim("wall", RegionKind.STUFF),)),
        _region("region-b", 20, 36, (_claim("wall", RegionKind.STUFF),)),
    )

    forward = reconcile_observation(regions, "frame-1", ReconciliationConfig()).entities
    backward = reconcile_observation(tuple(reversed(regions)), "frame-1", ReconciliationConfig()).entities

    assert forward[0].entity_id == backward[0].entity_id
    assert forward[0].member_region_ids == backward[0].member_region_ids


# Desligar a canonicalização preserva o comportamento anterior de label, o que
# torna a decisão ablatável em vez de embutida.
def test_canonicalization_can_be_disabled() -> None:
    """Com a canonicalização desligada, o conceito é apenas o label normalizado."""
    region = _region("region-a", 4, 20, (_claim("plain wall", RegionKind.STUFF),))

    result = reconcile_observation(
        (region,), "frame-1", ReconciliationConfig(canonicalize_labels=False)
    )

    assert result.records[0].canonical_concept == "plain wall"


# O estágio desligado é um caminho de custo zero explícito.
def test_a_disabled_stage_returns_the_regions_untouched() -> None:
    """Desligar a reconciliação devolve as regiões sem grupo nem claim nova."""
    regions = (_region("region-a", 4, 20, (_claim("plain wall", RegionKind.THING),)),)

    result = reconcile_observation(regions, "frame-1", ReconciliationConfig(enabled=False))

    assert result.regions is regions
    assert result.entities == ()
    assert result.records == ()


# ``region_concept`` é a única forma de perguntar "qual é a melhor
# interpretação atual", e ela devolve a reconciliada quando existe.
def test_the_region_concept_prefers_the_reconciled_claim() -> None:
    """O conceito de consumo é o reconciliado, com fallback para a primária."""
    region = _region("region-a", 4, 20, (_claim("plain wall", RegionKind.THING),))

    reconciled = reconcile_observation((region,), "frame-1", ReconciliationConfig()).regions[0]

    assert region_concept(reconciled) == "wall"
    assert region_concept(dataclasses.replace(region, claims=())) == ""
