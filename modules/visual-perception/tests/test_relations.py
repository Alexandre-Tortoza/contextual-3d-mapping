"""Testes de contract de relação candidata estruturada (#166)."""

from __future__ import annotations

import pytest

from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.relations import (
    CandidateRelation,
    RelationSource,
    validate_relation_references,
)
from visual_perception.domain.semantics import ConfidenceScore, Evidence


# Constrói uma CandidateRelation válida entre duas regiões nomeadas, com
# provenance/evidence mínimos, para servir de base aos testes de validação abaixo.
def _relation(subject: str, predicate: str, obj: str) -> CandidateRelation:
    return CandidateRelation(
        relation_id=f"rel-{subject}-{obj}",
        subject_region_id=subject,
        predicate=predicate,
        object_region_id=obj,
        confidence=ConfidenceScore(0.8, source="fake"),
        source=RelationSource.GEOMETRIC_2D,
        evidence=(Evidence("overlap"),),
        provenance=ModelProvenance(stage="test", producer="fake", config_fingerprint="abc"),
    )


# Uma relação bem formada entre duas regiões distintas deve ser aceita e preservar
# seu predicate.
def test_valid_relation() -> None:
    relation = _relation("region-a", "overlaps", "region-b")
    assert relation.predicate == "overlaps"


# Uma relação onde sujeito e objeto são a mesma região não faz sentido semântico e
# deve ser rejeitada na construção.
def test_rejects_self_relation() -> None:
    with pytest.raises(ValueError):
        _relation("region-a", "overlaps", "region-a")


# O predicate deve seguir uma convenção normalizada (ex: snake_case/lowercase); um
# predicate com espaços/capitalização livre é rejeitado para manter relações
# comparáveis e indexáveis.
def test_rejects_non_normalized_predicate() -> None:
    with pytest.raises(ValueError):
        _relation("region-a", "Overlaps With", "region-b")


# Toda relação candidata precisa de pelo menos uma Evidence que a sustente — uma
# relação sem evidência não é auditável e deve ser rejeitada.
def test_rejects_relation_without_evidence() -> None:
    with pytest.raises(ValueError):
        CandidateRelation(
            "rel-1",
            "region-a",
            "overlaps",
            "region-b",
            ConfidenceScore(0.8, source="fake"),
            RelationSource.GEOMETRIC_2D,
            (),
            ModelProvenance(stage="test", producer="fake", config_fingerprint="abc"),
        )


# validate_relation_references deve rejeitar uma relação que aponta para um
# region_id fora do conjunto de regiões conhecidas (referência pendurada/dangling).
def test_dangling_reference_is_rejected() -> None:
    relations = (_relation("region-a", "overlaps", "region-missing"),)
    with pytest.raises(ValueError):
        validate_relation_references(relations, frozenset({"region-a"}))


# Quando ambas as regiões referenciadas existem no conjunto conhecido, a validação
# deve passar silenciosamente (sem levantar exception).
def test_valid_reference_passes() -> None:
    relations = (_relation("region-a", "overlaps", "region-b"),)
    validate_relation_references(relations, frozenset({"region-a", "region-b"}))
