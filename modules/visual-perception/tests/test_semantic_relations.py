"""Testes da inferência de relações semânticas candidatas (#206)."""

from __future__ import annotations

import numpy as np
import pytest

from visual_perception.application.semantic_relations import (
    InvalidRelation,
    infer_semantic_relations,
    parse_relation_response,
    select_relation_candidates,
)
from visual_perception.config import MultimodalReasoningConfig, SemanticRelationConfig
from visual_perception.domain.contextual_entities import (
    ContextualEntityHypothesis,
    EntityHypothesisKind,
    EntityHypothesisStatus,
)
from visual_perception.domain.geometry import Mask
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.region_reasoning import RegionRelationRequest
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.relations import RelationSource
from visual_perception.domain.semantics import (
    ClaimKind,
    ConfidenceScore,
    Evidence,
    HypothesisRole,
    RegionKind,
    SemanticClaim,
)
from visual_perception.infrastructure.fakes.fake_multimodal_reasoner import FakeMultimodalReasoner

_PROVENANCE = ModelProvenance(stage="region_semantics", producer="fake", config_fingerprint="abc")
_REASONING = MultimodalReasoningConfig()


# Constrói uma claim de identidade reconciliada, que é o conceito que o estágio
# de relações usa ao formular a pergunta.
def _claim(value: str, kind: RegionKind = RegionKind.THING) -> SemanticClaim:
    return SemanticClaim(
        ClaimKind.LABEL,
        value,
        ConfidenceScore(0.9, source="fake"),
        (Evidence("raw region response"),),
        _PROVENANCE,
        role=HypothesisRole.PRIMARY,
        region_kind=kind,
    )


# Constrói uma região retangular na área pedida.
def _region(
    region_id: str, box: tuple[int, int, int, int], claims: tuple[SemanticClaim, ...]
) -> ObservedRegion:
    x_min, y_min, x_max, y_max = box
    data = np.zeros((64, 64), dtype=np.bool_)
    data[y_min:y_max, x_min:x_max] = True
    mask = Mask(data, 64, 64)
    return ObservedRegion(region_id, mask, mask.bounding_box(), 0.9, (f"{region_id}-p",), claims=claims)


# Constrói um payload de imagem simples do tamanho das máscaras acima.
def _payload() -> ImagePayload:
    pixels = np.full((64, 64, 3), 120, dtype=np.uint8)
    return ImagePayload(pixels, width=64, height=64)


# ``none`` é uma resposta válida e esperada, não uma falha. Sem essa saída, um
# vocabulário fechado empurra o modelo a escolher o predicado menos ruim, e a
# aresta inventada entra no grafo como se fosse observação.
def test_none_is_a_valid_answer() -> None:
    """Um predicado ``none`` é aceito e não vira relação."""
    assert parse_relation_response({"predicate": "none"}) == ("none", None)


# Os predicados do vocabulário são aceitos e normalizados, e a confiança
# ausente continua ausente.
@pytest.mark.parametrize("raw", ["part_of", "Part Of", "part-of", " PART_OF "])
def test_a_known_predicate_is_normalized(raw: str) -> None:
    """Grafias equivalentes do mesmo predicado colapsam na forma canônica."""
    assert parse_relation_response({"predicate": raw}) == ("part_of", None)


# Um predicado fora do vocabulário falha localmente em vez de virar aresta.
def test_an_unknown_predicate_is_rejected() -> None:
    """Um predicado fora do vocabulário invalida a resposta."""
    with pytest.raises(InvalidRelation, match="unknown predicate"):
        parse_relation_response({"predicate": "vibes_with"})


# As relações que exigem geometria 3D não estão no vocabulário, e uma resposta
# que as use é recusada: é a fronteira do módulo, imposta pelo parser.
@pytest.mark.parametrize("predicate", ["above", "below", "behind", "in_front_of"])
def test_predicates_that_require_3d_geometry_are_rejected(predicate: str) -> None:
    """Relações que dependem de profundidade não são inferíveis de um frame."""
    with pytest.raises(InvalidRelation):
        parse_relation_response({"predicate": predicate})


# Predicados que o caminho geométrico já responde exatamente ficam fora do
# vocabulário semântico. ``inside`` saiu por medição: com a fração de contenção
# no prompt o modelo devolveu ``inside`` em 6 de 16 pares; sem ela, em 0 de 16.
# Ele repetia o número, não lia os pixels.
@pytest.mark.parametrize("predicate", ["near", "adjacent_to", "inside", "contains"])
def test_predicates_the_geometric_channel_already_answers_are_rejected(predicate: str) -> None:
    """O canal semântico não duplica uma pergunta que a geometria já mede."""
    with pytest.raises(InvalidRelation):
        parse_relation_response({"predicate": predicate})


# O resumo geométrico entregue ao modelo não pode conter a resposta: ele fala de
# contato e tamanho relativo, e nenhum dos dois é sinônimo de um predicado.
def test_the_geometric_summary_does_not_hand_over_a_predicate() -> None:
    """O resumo não menciona contenção, que é sinônimo de um predicado removido."""
    container = _region("region-a", (4, 4, 40, 40), (_claim("cabinet"),))
    part = _region("region-b", (8, 8, 20, 20), (_claim("handle"),))

    summary = select_relation_candidates((part, container), SemanticRelationConfig())[0].geometric_summary()

    assert "covers" not in summary
    assert "%" not in summary
    assert "larger" in summary


# Confiança ausente permanece ausente; um valor fora de [0, 1] é rejeitado.
def test_confidence_is_optional_and_validated() -> None:
    """A confiança do produtor é opcional e nunca é inventada."""
    assert parse_relation_response({"predicate": "covers", "confidence": 0.4}) == ("covers", 0.4)
    assert parse_relation_response({"predicate": "covers"})[1] is None
    with pytest.raises(InvalidRelation):
        parse_relation_response({"predicate": "covers", "confidence": 1.5})
    with pytest.raises(InvalidRelation):
        parse_relation_response({"predicate": "covers", "confidence": "high"})


# A seleção prioriza contenção, que é o sinal geométrico mais forte de que há
# uma relação real, e o sujeito é o lado que contém: sem essa regra, "A contém
# B" e "B está dentro de A" viram duas perguntas sobre a mesma evidência.
def test_containment_is_prioritized_and_orients_the_pair() -> None:
    """O par contido é priorizado, com o continente como sujeito."""
    container = _region("region-a", (4, 4, 40, 40), (_claim("cabinet"),))
    part = _region("region-b", (8, 8, 20, 20), (_claim("handle"),))

    candidates = select_relation_candidates((part, container), SemanticRelationConfig())

    assert len(candidates) == 1
    assert candidates[0].subject.region_id == "region-a"
    assert candidates[0].target.region_id == "region-b"
    assert candidates[0].containment == pytest.approx(1.0)


# Pares dentro do mesmo grupo de superfície são descartados: a reconciliação já
# descreve a relação entre eles melhor do que uma aresta descreveria.
def test_pairs_inside_a_same_surface_group_are_skipped() -> None:
    """Membros do mesmo grupo não geram par candidato."""
    first = _region("region-a", (4, 4, 20, 20), (_claim("wall", RegionKind.STUFF),))
    second = _region("region-b", (20, 4, 36, 20), (_claim("panel", RegionKind.STUFF),))
    entity = ContextualEntityHypothesis(
        entity_id="entity-abc",
        kind=EntityHypothesisKind.SAME_SURFACE,
        canonical_concept="wall",
        region_kind=RegionKind.STUFF,
        member_region_ids=("region-a", "region-b"),
        status=EntityHypothesisStatus.SUPPORTED,
        evidence=(Evidence("touching regions share a concept"),),
        provenance=_PROVENANCE,
    )

    assert select_relation_candidates((first, second), SemanticRelationConfig(), entities=(entity,)) == ()


# Dois recortes do mesmo conceito são fragmentação, não relação: perguntar
# "como esta parede se relaciona com esta parede" gasta chamada e produz ruído.
def test_pairs_with_the_same_concept_are_skipped_by_default() -> None:
    """Pares de conceito idêntico são descartados quando a política pede isso."""
    first = _region("region-a", (4, 4, 20, 20), (_claim("wall", RegionKind.STUFF),))
    second = _region("region-b", (20, 4, 36, 20), (_claim("wall", RegionKind.STUFF),))

    assert select_relation_candidates((first, second), SemanticRelationConfig()) == ()
    assert select_relation_candidates(
        (first, second), SemanticRelationConfig(require_distinct_concepts=False)
    )


# O orçamento é um teto duro, e a ordem sob teto é determinística: sem ele o
# estágio é O(n²) chamadas de VLM.
def test_the_pair_budget_is_a_hard_ceiling_and_deterministic() -> None:
    """A seleção respeita ``max_pairs`` e é estável entre execuções."""
    regions = tuple(
        _region(f"region-{index}", (index * 6, 4, index * 6 + 8, 20), (_claim(f"thing-{index}"),))
        for index in range(6)
    )
    config = SemanticRelationConfig(max_pairs=3)

    first = select_relation_candidates(regions, config)
    second = select_relation_candidates(regions, config)

    assert len(first) == 3
    assert [(c.subject.region_id, c.target.region_id) for c in first] == [
        (c.subject.region_id, c.target.region_id) for c in second
    ]


# O caminho feliz de ponta a ponta: uma relação inferida vira uma
# ``CandidateRelation`` MODEL_INFERRED, com a resposta bruta preservada.
def test_an_inferred_relation_keeps_its_source_and_raw_evidence() -> None:
    """Uma relação inferida preserva fonte, proveniência e resposta bruta."""
    container = _region("region-a", (4, 4, 40, 40), (_claim("cabinet"),))
    part = _region("region-b", (8, 8, 20, 20), (_claim("handle"),))
    reasoner = FakeMultimodalReasoner(
        relation_response_fn=lambda request: {"predicate": "part_of", "confidence": 0.55}
    )

    result = infer_semantic_relations(
        (part, container), _payload(), reasoner, SemanticRelationConfig(), _REASONING
    )

    assert len(result.relations) == 1
    relation = result.relations[0]
    assert relation.source is RelationSource.MODEL_INFERRED
    assert relation.predicate == "part_of"
    assert relation.subject_region_id == "region-a"
    assert relation.object_region_id == "region-b"
    assert relation.confidence is not None
    assert relation.confidence.value == pytest.approx(0.55)
    assert relation.evidence[0].raw_response_json is not None
    assert relation.provenance.stage == "semantic_relations"
    assert result.model_calls == 1


# Um produtor que não pontua produz uma relação sem score, nunca uma com 1.0.
def test_an_unscored_relation_stays_unscored() -> None:
    """Ausência de score sobrevive à inferência de relação."""
    container = _region("region-a", (4, 4, 40, 40), (_claim("cabinet"),))
    part = _region("region-b", (8, 8, 20, 20), (_claim("handle"),))
    reasoner = FakeMultimodalReasoner(relation_response_fn=lambda request: {"predicate": "covers"})

    result = infer_semantic_relations(
        (part, container), _payload(), reasoner, SemanticRelationConfig(), _REASONING
    )

    assert result.relations[0].confidence is None


# Uma resposta malformada derruba o par, e só ele: os demais continuam.
def test_a_malformed_response_fails_locally() -> None:
    """Uma resposta inválida vira falha isolada, sem invalidar os outros pares."""
    regions = (
        _region("region-a", (4, 4, 40, 40), (_claim("cabinet"),)),
        _region("region-b", (8, 8, 20, 20), (_claim("handle"),)),
        _region("region-c", (44, 4, 60, 20), (_claim("lamp"),)),
    )

    def respond(request: RegionRelationRequest) -> dict[str, object]:
        if request.object_region_id == "region-b":
            return {"predicate": "levitates_over"}
        return {"predicate": "supported_by"}

    result = infer_semantic_relations(
        regions, _payload(), FakeMultimodalReasoner(relation_response_fn=respond),
        SemanticRelationConfig(), _REASONING
    )

    assert len(result.failures) == 1
    assert result.failures[0].object_region_id == "region-b"
    assert all(relation.predicate == "supported_by" for relation in result.relations)


# Toda relação produzida referencia regiões conhecidas e nunca é auto-relação:
# o contract de ``CandidateRelation`` recusaria, e a seleção nunca as forma.
def test_every_inferred_relation_references_two_distinct_known_regions() -> None:
    """Nenhuma relação inferida é auto-relação nem referencia região desconhecida."""
    regions = (
        _region("region-a", (4, 4, 40, 40), (_claim("cabinet"),)),
        _region("region-b", (8, 8, 20, 20), (_claim("handle"),)),
    )
    reasoner = FakeMultimodalReasoner(relation_response_fn=lambda request: {"predicate": "part_of"})

    result = infer_semantic_relations(
        regions, _payload(), reasoner, SemanticRelationConfig(), _REASONING
    )

    known = {region.region_id for region in regions}
    for relation in result.relations:
        assert relation.subject_region_id != relation.object_region_id
        assert {relation.subject_region_id, relation.object_region_id} <= known


# O estágio desligado é um caminho de custo zero explícito.
def test_a_disabled_stage_makes_no_calls() -> None:
    """Desligar o estágio não consulta o reasoner nem produz relação."""
    regions = (
        _region("region-a", (4, 4, 40, 40), (_claim("cabinet"),)),
        _region("region-b", (8, 8, 20, 20), (_claim("handle"),)),
    )

    result = infer_semantic_relations(
        regions, _payload(), FakeMultimodalReasoner(), SemanticRelationConfig(enabled=False), _REASONING
    )

    assert result.relations == ()
    assert result.model_calls == 0
